from typing import Optional, List
from dataclasses import dataclass
import tqdm
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from transformers.modeling_outputs import ModelOutput

from src.core.sbrec import SBRec
from src.core.bert4rec import BERT4Rec
from src.core.rec_metrics import (
    _compute_apk,
    _compute_precision_recall,
    _compute_ndcg,
    _compute_hr,
)


@dataclass
class SequentialRecommendationOutput(ModelOutput):
    predicts: Optional[List[List[float]]] = None
    loss: Optional[torch.FloatTensor] = None
    recommend_loss: Optional[torch.FloatTensor] = None
    rating_loss: Optional[torch.FloatTensor] = None


class SequentialRecommendation(nn.Module):
    def __init__(
        self,
        num_items,
        sbrec_embedding_dim=384,
        sbrec_num_attention_heads=4,
        sbrec_use_pos_emb=False,
        bert4rec_embedding_dim=64,
        bert4rec_num_layers=4,
        bert4rec_num_heads=4,
        bert4rec_hidden_dim=256,
        use_rating_loss=False,
    ):
        super().__init__()
        self.sbrec_model = SBRec(
            embedding_dim=sbrec_embedding_dim,
            num_attention_heads=sbrec_num_attention_heads,
            use_pos_emb=sbrec_use_pos_emb,
            max_seq_length=50,
        )
        self.bert4rec_model = BERT4Rec(
            num_items=num_items,
            embedding_dim=bert4rec_embedding_dim,
            num_layers=bert4rec_num_layers,
            num_heads=bert4rec_num_heads,
            hidden_dim=bert4rec_hidden_dim,
            dropout=0.1,
            max_seq_length=50,
        )
        self.use_rating_loss = use_rating_loss
        if use_rating_loss:
            self.rating_predictor = nn.Linear(
                2 * (self.sbrec_model.embedding_dim + bert4rec_embedding_dim), 1
            )

    def forward(
        self,
        sequences,
        targets,
        targets_rating,
        targets_mask,
        negative_targets,
        negative_targets_mask,
        sequences_text_embedding,
        sequences_text_embedding_mask,
        targets_text_embedding,
        targets_text_embedding_mask,
        negative_targets_text_embedding,
        negative_targets_text_embedding_mask,
    ) -> SequentialRecommendationOutput:
        # text embedding branch
        sequences_text_embedding = self.sbrec_model.forward(
            sequences_text_embedding,
            sequences_text_embedding_mask,
            output_type="sequence",
        )
        targets_text_embedding = self.sbrec_model.forward(
            targets_text_embedding, targets_text_embedding_mask, output_type="item"
        )
        negative_targets_text_embedding = self.sbrec_model.forward(
            negative_targets_text_embedding,
            negative_targets_text_embedding_mask,
            output_type="item",
        )

        # id embedding branch
        sequences_id_embedding = self.bert4rec_model.forward(
            sequences, output_type="sequence"
        )
        targets_id_embedding = self.bert4rec_model.forward(targets, output_type="item")
        negative_targets_id_embedding = self.bert4rec_model.forward(
            negative_targets, output_type="item"
        )

        # concat sequences
        # torch concat sequences_text_embedding (shape B x text_embedding_dim) and sequences_id_embedding (shape B x id_embedding_dim)
        # output shape B x (text_embedding_dim+id_embedding_dim)
        sequences_embedding = torch.cat(
            [sequences_text_embedding, sequences_id_embedding], dim=-1
        )

        # concat targets
        # torch concat targets_text_embedding (shape B x target_length x text_embedding_dim) and targets_id_embedding (shape B x target_length x id_embedding_dim)
        # output shape B x target_length x (text_embedding_dim+id_embedding_dim)
        targets_embedding = torch.cat(
            [targets_text_embedding, targets_id_embedding], dim=-1
        )

        # concat negative_targets
        # torch concat negative_targets_text_embedding (shape B x n_neg_samples x text_embedding_dim) and negative_targets_id_embedding (shape B x n_neg_samples x id_embedding_dim)
        # output shape B x n_neg_samples x (text_embedding_dim+id_embedding_dim)
        negative_targets_embedding = torch.cat(
            [negative_targets_text_embedding, negative_targets_id_embedding], dim=-1
        )

        # compute loss
        # compute score between sequences_embedding and targets_embedding
        # score shape B x target_length
        score_targets = torch.bmm(
            targets_embedding, sequences_embedding.unsqueeze(-1)
        ).squeeze(-1)  # B x target_length
        # compute score between sequences_embedding and negative_targets_embedding
        # score shape B x n_neg_samples
        score_negative_targets = torch.bmm(
            negative_targets_embedding, sequences_embedding.unsqueeze(-1)
        ).squeeze(-1)  # B x n_neg_samples

        # compute loss
        # loss = -sum(log(sigmoid(score_targets))) - sum(log(sigmoid(-score_negative_targets)))
        loss_targets = -torch.log(torch.sigmoid(score_targets)) * targets_mask
        loss_negative_targets = (
            -torch.log(torch.sigmoid(-score_negative_targets)) * negative_targets_mask
        )
        recommend_loss = torch.sum(loss_targets) + torch.sum(loss_negative_targets)
        recommend_loss = recommend_loss / (
            torch.sum(targets_mask) + torch.sum(negative_targets_mask)
        )
        loss = recommend_loss

        # compute predict
        predicts = []  # B list item_id (except 0 (padding_id))
        item_ids = torch.cat([targets, negative_targets], dim=-1).detach().cpu().numpy()
        scores = (
            torch.cat([score_targets, score_negative_targets], dim=-1)
            .detach()
            .cpu()
            .numpy()
        )
        for i in range(item_ids.shape[0]):
            item_ids_ = item_ids[i]
            scores_ = scores[i]
            predicts_ = item_ids_[scores_.argsort()[::-1]].tolist()
            predicts_ = [p for p in predicts_ if p > 0]
            predicts.append(predicts_)

        # get rating loss
        if self.use_rating_loss:
            # concat sequences_embedding and targets_embedding
            # sequences_embedding shape: B x (text_embedding_dim+id_embedding_dim)
            # targets_embedding shape: B x target_length x (text_embedding_dim+id_embedding_dim)
            # concat shape --> B x target_length x (text_embedding_dim+id_embedding_dim)
            sequences_embedding_expanded = sequences_embedding.unsqueeze(1).expand_as(
                targets_embedding
            )
            combined_embedding = torch.cat(
                [sequences_embedding_expanded, targets_embedding], dim=-1
            )

            # combined_embedding shape: B x target_length x (2 * (text_embedding_dim+id_embedding_dim))
            # linear layer output shape: B x target_length x 1
            predicted_rating = self.rating_predictor(combined_embedding).squeeze(
                -1
            )  # B x target_length

            # compute rating loss
            # use mean squared error
            rating_loss = F.mse_loss(
                predicted_rating * targets_mask,
                targets_rating * targets_mask,
                reduction="mean",
            )
            rating_loss = rating_loss / torch.sum(targets_mask)

            # combine losses
            loss = loss + rating_loss * 10
        else:
            rating_loss = None

        return SequentialRecommendationOutput(
            predicts=predicts,
            loss=loss,
            recommend_loss=recommend_loss,
            rating_loss=rating_loss,
        )

    def pre_epoch(self, trainer, epoch):
        # do nothing
        pass

    def train_epoch(self, trainer, epoch):
        self.train()

        # init
        log_info = {"step_losses": [], "epoch_loss": None, "others": {}}
        recommend_losses = []
        rating_losses = []
        n_true_pred = 0
        n_false_pred = 0

        # loop
        for batch_data in tqdm.tqdm(
            trainer.dataloader_train, ncols=100, desc=f"Train epoch {epoch}"
        ):
            # get input (.to(self.device))
            # user_id = batch_data["user_id"].to(self.device)  # B --> not use
            sequences = batch_data["sequences"].to(self.device)  # B x sequence_length
            targets = batch_data["targets"].to(self.device)  # B x max(target_length)
            targets_rating = batch_data["targets_rating"].to(
                self.device
            )  # B x max(target_length)
            targets_mask = batch_data["targets_mask"].to(
                self.device
            )  # B x max(target_length)
            negative_targets = batch_data["negative_targets"].to(
                self.device
            )  # B x max(n_neg_samples)
            negative_targets_mask = batch_data["negative_targets_mask"].to(
                self.device
            )  # B x max(n_neg_samples)
            sequences_text_embedding = batch_data["sequences_text_embedding"].to(
                self.device
            )  # B x sequence_length x max(n_review) x embedding_dim
            sequences_text_embedding_mask = batch_data[
                "sequences_text_embedding_mask"
            ].to(self.device)  # B x sequence_length x max(n_review)
            targets_text_embedding = batch_data["targets_text_embedding"].to(
                self.device
            )  # B x max(target_length) x max(n_review) x embedding_dim
            targets_text_embedding_mask = batch_data["targets_text_embedding_mask"].to(
                self.device
            )  # B x max(target_length) x max(n_review)
            negative_targets_text_embedding = batch_data[
                "negative_targets_text_embedding"
            ].to(self.device)  # B x max(n_neg_samples) x max(n_review) x embedding_dim
            negative_targets_text_embedding_mask = batch_data[
                "negative_targets_text_embedding_mask"
            ].to(self.device)  # B x max(n_neg_samples) x max(n_review)

            # forward
            output = self.forward(
                sequences,
                targets,
                targets_rating,
                targets_mask,
                negative_targets,
                negative_targets_mask,
                sequences_text_embedding,
                sequences_text_embedding_mask,
                targets_text_embedding,
                targets_text_embedding_mask,
                negative_targets_text_embedding,
                negative_targets_text_embedding_mask,
            )

            # backward
            loss = output.loss
            loss.backward()
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()

            # log step
            log_info["step_losses"].append(loss.item())
            recommend_losses.append(output.recommend_loss.item())
            rating_losses.append(
                0 if output.rating_loss is None else output.rating_loss.item()
            )

            # get acc
            if len(output.predicts) > 0:
                for i in range(len(output.predicts)):
                    if output.predicts[i][0] in targets[i]:
                        n_true_pred += 1
                    else:
                        n_false_pred += 1

        # log epoch
        log_info["epoch_loss"] = np.mean(log_info["step_losses"])
        log_info["others"]["acc"] = float(n_true_pred) / (n_true_pred + n_false_pred)
        log_info["others"]["recommend_loss"] = np.mean(recommend_losses)
        log_info["others"]["rating_loss"] = np.mean(rating_losses)

        return log_info

    def test_epoch(self, trainer, dataloader, epoch, test_name):
        self.eval()

        # init
        log_info = {}
        step_losses = []
        all_precision = {"1": [], "5": [], "10": []}
        all_recall = {"1": [], "5": [], "10": []}
        all_f1 = {"1": [], "5": [], "10": []}
        all_ndcg = {"1": [], "5": [], "10": []}
        all_hr = {"1": [], "5": [], "10": []}
        all_apk = {"1": [], "5": [], "10": []}
        recommend_losses = []
        rating_losses = []

        # loop
        for batch_data in tqdm.tqdm(
            dataloader, ncols=100, desc=f"Test {test_name}_data epoch {epoch}"
        ):
            # get input (.to(self.device))
            # user_id = batch_data["user_id"].to(self.device)  # B --> not use
            sequences = batch_data["sequences"].to(self.device)  # B x sequence_length
            targets = batch_data["targets"].to(self.device)  # B x max(target_length)
            targets_rating = batch_data["targets_rating"].to(
                self.device
            )  # B x max(target_length)
            targets_mask = batch_data["targets_mask"].to(
                self.device
            )  # B x max(target_length)
            negative_targets = batch_data["negative_targets"].to(
                self.device
            )  # B x max(n_neg_samples)
            negative_targets_mask = batch_data["negative_targets_mask"].to(
                self.device
            )  # B x max(n_neg_samples)
            sequences_text_embedding = batch_data["sequences_text_embedding"].to(
                self.device
            )  # B x sequence_length x max(n_review) x embedding_dim
            sequences_text_embedding_mask = batch_data[
                "sequences_text_embedding_mask"
            ].to(self.device)  # B x sequence_length x max(n_review)
            targets_text_embedding = batch_data["targets_text_embedding"].to(
                self.device
            )  # B x max(target_length) x max(n_review) x embedding_dim
            targets_text_embedding_mask = batch_data["targets_text_embedding_mask"].to(
                self.device
            )  # B x max(target_length) x max(n_review)
            negative_targets_text_embedding = batch_data[
                "negative_targets_text_embedding"
            ].to(self.device)  # B x max(n_neg_samples) x max(n_review) x embedding_dim
            negative_targets_text_embedding_mask = batch_data[
                "negative_targets_text_embedding_mask"
            ].to(self.device)  # B x max(n_neg_samples) x max(n_review)

            # forward
            with torch.no_grad():
                try:
                    output = self.forward(
                        sequences,
                        targets,
                        targets_rating,
                        targets_mask,
                        negative_targets,
                        negative_targets_mask,
                        sequences_text_embedding,
                        sequences_text_embedding_mask,
                        targets_text_embedding,
                        targets_text_embedding_mask,
                        negative_targets_text_embedding,
                        negative_targets_text_embedding_mask,
                    )
                except:
                    # skip sample out of mem
                    print(
                        "skip sample out of mem:", negative_targets_text_embedding.shape
                    )

            # log step
            step_losses.append(output.loss.item())
            recommend_losses.append(output.recommend_loss.item())
            rating_losses.append(
                0 if output.rating_loss is None else output.rating_loss.item()
            )

            # get score info
            targets = targets.detach().cpu().numpy().tolist()
            targets = [[i for i in target if i > 0] for target in targets]
            predicts = output.predicts
            for k in [1, 5, 10]:
                for i in range(len(targets)):
                    precision, recall = _compute_precision_recall(
                        targets[i], predicts[i], k
                    )
                    if precision != 0 or recall != 0:
                        f1 = 2.0 * precision * recall / (precision + recall)
                    else:
                        f1 = 0.0
                    ndcg = _compute_ndcg(targets[i], predicts[i], k)
                    hr = _compute_hr(targets[i], predicts[i], k)
                apk = _compute_apk(targets[i], predicts[i], k=np.inf)

                all_precision[str(k)].append(precision)
                all_recall[str(k)].append(recall)
                all_f1[str(k)].append(f1)
                all_ndcg[str(k)].append(ndcg)
                all_hr[str(k)].append(hr)
                all_apk[str(k)].append(apk)

        # log info
        log_info["loss"] = np.mean(step_losses)
        log_info["recommend_loss"] = np.mean(recommend_losses)
        log_info["rating_loss"] = np.mean(rating_losses)
        # for k in [1, 5, 10]:
        for k in [5]:
            log_info[f"precision@{k}"] = np.mean(all_precision[str(k)])
            log_info[f"recall@{k}"] = np.mean(all_recall[str(k)])
            log_info[f"f1@{k}"] = np.mean(all_f1[str(k)])
            log_info[f"ndcg@{k}"] = np.mean(all_ndcg[str(k)])
            log_info[f"hr@{k}"] = np.mean(all_hr[str(k)])
            log_info[f"apk@{k}"] = np.mean(all_apk[str(k)])

        # log epoch
        log_info["epoch_loss"] = np.mean(step_losses)

        return log_info

    def post_epoch(self, trainer, epoch):
        # do nothing
        pass

    def save(self, save_path):
        torch.save(self.state_dict(), save_path + ".pth")
