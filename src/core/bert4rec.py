import torch
import torch.nn as nn
import torch.nn.functional as F


class BERT4Rec(nn.Module):
    def __init__(
        self,
        num_items,
        embedding_dim,
        num_layers,
        num_heads,
        hidden_dim,
        dropout=0.1,
        max_seq_length=50,
    ):
        """
        Khởi tạo mô hình BERT4Rec.

        Args:
            num_items (int): Tổng số lượng item.
            embedding_dim (int): Kích thước của embedding vector.
            num_layers (int): Số lượng lớp Transformer.
            num_heads (int): Số lượng attention head trong Multi-Head Attention.
            hidden_dim (int): Kích thước lớp ẩn trong Feed-Forward Network.
            dropout (float): Tỷ lệ dropout.
            max_seq_length (int): Độ dài tối đa của sequence. (dùng cho position embedding)
        """
        super(BERT4Rec, self).__init__()

        self.embedding = nn.Embedding(
            num_items + 1, embedding_dim, padding_idx=0
        )  # +1 cho item padding, padding_idx=0
        self.pos_embedding = nn.Embedding(max_seq_length, embedding_dim)
        self.transformer_layers = nn.ModuleList(
            [
                TransformerLayer(embedding_dim, num_heads, hidden_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.max_seq_length = max_seq_length
        self.init_weights()

    def init_weights(self):
        """Khởi tạo weights của các lớp."""
        for name, param in self.named_parameters():
            if "embedding" in name:
                nn.init.normal_(param, std=0.02)
            elif "layer_norm" in name:
                nn.init.constant_(param, 1)
            elif "bias" in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def forward(self, sequences, output_type):
        """
        Forward pass của mô hình.

        Args:
            sequences (torch.Tensor): Input sequence (B x seq_length).

        Returns:
            torch.Tensor: Output embedding (B x embedding_dim).
        """
        batch_size, seq_length = sequences.size()

        # 1. Embedding
        item_embeddings = self.embedding(sequences)  # (B x seq_length x embedding_dim)
        if output_type == "item":
            return item_embeddings
        elif output_type == "sequence":
            positions = (
                torch.arange(seq_length, device=sequences.device)
                .unsqueeze(0)
                .expand(batch_size, seq_length)
            )
            pos_embeddings = self.pos_embedding(
                positions
            )  # (B x seq_length x embedding_dim)
            embeddings = item_embeddings + pos_embeddings
            embeddings = self.layer_norm(embeddings)
            embeddings = self.dropout(embeddings)

            # 2. Transformer Layers
            for layer in self.transformer_layers:
                embeddings = layer(embeddings)

            # 3. Output Layer
            # Lấy embedding của item đầu tiên trong sequence
            first_token_tensor = embeddings[:, 0, :]  # (B x embedding_dim)

            return first_token_tensor
        else:
            raise Exception("err with output_type")


class TransformerLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(embedding_dim, hidden_dim, dropout)
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass của một lớp Transformer.

        Args:
            x (torch.Tensor): Input tensor (B x seq_length x embedding_dim).

        Returns:
            torch.Tensor: Output tensor (B x seq_length x embedding_dim).
        """
        # Multi-Head Attention
        attention_output = self.attention(x, x, x)
        x = x + self.dropout(attention_output)
        x = self.layer_norm1(x)

        # Feed-Forward Network
        feed_forward_output = self.feed_forward(x)
        x = x + self.dropout(feed_forward_output)
        x = self.layer_norm2(x)

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(embedding_dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(embedding_dim, self.all_head_size)
        self.key = nn.Linear(embedding_dim, self.all_head_size)
        self.value = nn.Linear(embedding_dim, self.all_head_size)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(
            embedding_dim, embedding_dim
        )  # Thêm lớp linear projection sau attention

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value):
        """
        Forward pass của Multi-Head Attention.

        Args:
            query (torch.Tensor): Query tensor (B x seq_length x embedding_dim).
            key (torch.Tensor): Key tensor (B x seq_length x embedding_dim).
            value (torch.Tensor): Value tensor (B x seq_length x embedding_dim).

        Returns:
            torch.Tensor: Output tensor (B x seq_length x embedding_dim).
        """
        # Linear transformations
        mixed_query_layer = self.query(query)
        mixed_key_layer = self.key(key)
        mixed_value_layer = self.value(value)

        # Split into heads
        query_layer = self.transpose_for_scores(
            mixed_query_layer
        )  # (B x num_heads x seq_length x attention_head_size)
        key_layer = self.transpose_for_scores(
            mixed_key_layer
        )  # (B x num_heads x seq_length x attention_head_size)
        value_layer = self.transpose_for_scores(
            mixed_value_layer
        )  # (B x num_heads x seq_length x attention_head_size)

        # Attention scores
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2)
        )  # (B x num_heads x seq_length x seq_length)
        attention_scores = attention_scores / (self.attention_head_size**0.5)

        # Attention probabilities
        attention_probs = nn.Softmax(dim=-1)(
            attention_scores
        )  # (B x num_heads x seq_length x seq_length)
        attention_probs = self.dropout(attention_probs)

        # Context vector
        context_layer = torch.matmul(
            attention_probs, value_layer
        )  # (B x num_heads x seq_length x attention_head_size)
        context_layer = context_layer.permute(
            0, 2, 1, 3
        ).contiguous()  # (B x seq_length x num_heads x attention_head_size)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(
            *new_context_layer_shape
        )  # (B x seq_length x embedding_dim)
        context_layer = self.out_proj(context_layer)  # linear projection

        return context_layer


class PositionwiseFeedForward(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass của Position-wise Feed-Forward Network.

        Args:
            x (torch.Tensor): Input tensor (B x seq_length x embedding_dim).

        Returns:
            torch.Tensor: Output tensor (B x seq_length x embedding_dim).
        """
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x
