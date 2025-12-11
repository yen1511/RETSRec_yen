import torch
from torch import nn


class SBRec(nn.Module):
    def __init__(
        self, embedding_dim, num_attention_heads=4, use_pos_emb=False, max_seq_length=50
    ):
        super(SBRec, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.use_pos_emb = use_pos_emb

        if self.use_pos_emb:
            self.pos_embedding = nn.Embedding(max_seq_length, embedding_dim)

        # Bước 1: Item Embedding từ Review Embeddings
        self.item_attention = nn.MultiheadAttention(embedding_dim, num_attention_heads)

        # Bước 2: Self-Attention cho Sequence
        self.sequence_attention = nn.MultiheadAttention(
            embedding_dim, num_attention_heads
        )

        # Bước 3: Sequence Embedding từ Item Embeddings
        self.final_attention = nn.MultiheadAttention(embedding_dim, num_attention_heads)

    def forward(
        self, sequences_text_embedding, sequences_text_embedding_mask, output_type
    ):
        """
        Args:
            sequences_text_embedding: B x max(sequence_length) x max(n_review) x embedding_dim
            sequences_text_embedding_mask: B x max(sequence_length) x max(n_review)

        Returns:
            sequence_embedding: B x embedding_dim
        """
        # Bước 1: Item Embedding
        # Chuyển vị để phù hợp với MultiheadAttention (sequence_length, batch_size, embedding_dim)
        B, seq_len, num_reviews, emb_dim = sequences_text_embedding.shape
        item_embeddings = sequences_text_embedding.permute(1, 0, 2, 3).reshape(
            seq_len * B, num_reviews, emb_dim
        )
        item_mask = sequences_text_embedding_mask.reshape(seq_len * B, num_reviews)

        # item_embeddings: (seq_len * B) x max(n_review) x embedding_dim
        # item_mask: (seq_len * B)  x max(n_review)
        item_embeddings = item_embeddings.permute(
            1, 0, 2
        )  # max(n_review) x (seq_len * B) x embedding_dim
        item_embedding, _ = self.item_attention(
            item_embeddings,
            item_embeddings,
            item_embeddings,
            key_padding_mask=(1 - item_mask).bool(),
        )
        # item_embedding: max(n_review) x (seq_len * B) x embedding_dim
        item_embedding = item_embedding[0]  #  (seq_len * B) x embedding_dim
        item_embedding = item_embedding.reshape(seq_len, B, emb_dim).permute(
            1, 0, 2
        )  # B x seq_len x embedding_dim

        if output_type == "item":
            return item_embedding
        elif output_type == "sequence":
            if self.use_pos_emb:
                batch_size, seq_length, embedding_dim = item_embedding.size()
                positions = (
                    torch.arange(seq_length, device=item_embedding.device)
                    .unsqueeze(0)
                    .expand(batch_size, seq_length)
                )
                pos_embeddings = self.pos_embedding(
                    positions
                )  # (B x seq_length x embedding_dim)
                item_embedding = item_embedding + pos_embeddings

            # get mask
            sequence_mask = sequences_text_embedding_mask[:, :, 0]  # B x max(seq_len)

            # Bước 2: Self-Attention cho Sequence
            # sequence_embeddings: B x seq_len x embedding_dim
            sequence_embeddings = item_embedding.permute(
                1, 0, 2
            )  # max(seq_len) x B x embedding_dim
            sequence_embeddings, _ = self.sequence_attention(
                sequence_embeddings,
                sequence_embeddings,
                sequence_embeddings,
                key_padding_mask=(1 - sequence_mask).bool(),
            )
            sequence_embeddings = sequence_embeddings.permute(
                1, 0, 2
            )  # B x max(seq_len) x embedding_dim

            # Bước 3: Sequence Embedding
            sequence_embeddings = sequence_embeddings.permute(
                1, 0, 2
            )  # max(seq_len) x B x embedding_dim
            sequence_embedding, _ = self.final_attention(
                sequence_embeddings,
                sequence_embeddings,
                sequence_embeddings,
                key_padding_mask=(1 - sequence_mask).bool(),
            )
            sequence_embedding = sequence_embedding[0]  # B x embedding_dim

            return sequence_embedding
        else:
            raise Exception("err with output_type")
