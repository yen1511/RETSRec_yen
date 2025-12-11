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


class SequentialRecommendationOld(nn.Module):
    def __init__(
        self,
        num_items,
        sbrec_embedding_dim=384,
        sbrec_num_attention_heads=4,
    ):
        super().__init__()
        self.sbrec_model_user = SBRec(
            embedding_dim=sbrec_embedding_dim,
            num_attention_heads=sbrec_num_attention_heads,
            use_pos_emb=False,
            max_seq_length=50,
        )
        self.sbrec_model_item = SBRec(
            embedding_dim=sbrec_embedding_dim,
            num_attention_heads=sbrec_num_attention_heads,
            use_pos_emb=False,
            max_seq_length=50,
        )
        self.rating_predictor = nn.Linear(
            (self.sbrec_model_user.embedding_dim + self.sbrec_model_item.embedding_dim),
            1,
        )

    def forward(
        self,
        targets,
        targets_rating,
        targets_mask,
        negative_targets,
        negative_targets_mask,
        user_embedding,
        user_embedding_mask,
        targets_embedding,
        targets_embedding_mask,
        negative_targets_embedding,
        negative_targets_embedding_mask,
    ) -> SequentialRecommendationOutput:
        user_embedding = self.sbrec_model_user.forward(
            user_embedding,
            user_embedding_mask,
            output_type="item",
        )
        user_embedding = user_embedding[:, 0, :]
        targets_embedding = self.sbrec_model_item.forward(
            targets_embedding,
            targets_embedding_mask,
            output_type="item",
        )
        negative_targets_embedding = self.sbrec_model_item.forward(
            negative_targets_embedding,
            negative_targets_embedding_mask,
            output_type="item",
        )

        # concat
        user_embedding_expanded = user_embedding.unsqueeze(1).expand_as(
            targets_embedding
        )
        score_targets = self.rating_predictor(
            torch.cat([user_embedding_expanded, targets_embedding], dim=-1)
        ).squeeze(-1)
        user_embedding_expanded = user_embedding.unsqueeze(1).expand_as(
            negative_targets_embedding
        )
        score_negative_targets = self.rating_predictor(
            torch.cat([user_embedding_expanded, negative_targets_embedding], dim=-1)
        ).squeeze(-1)

        # compute loss
        loss = F.mse_loss(
            score_targets * targets_mask,
            targets_rating * targets_mask,
            reduction="mean",
        )
        loss = loss / torch.sum(targets_mask)

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

        return SequentialRecommendationOutput(
            predicts=predicts,
            loss=loss,
        )

    def pre_epoch(self, trainer, epoch):
        # do nothing
        pass

    def train_epoch(self, trainer, epoch):
        self.train()

        # init
        log_info = {"step_losses": [], "epoch_loss": None, "others": {}}
        n_true_pred = 0
        n_false_pred = 0

        # loop
        for batch_data in tqdm.tqdm(
            trainer.dataloader_train, ncols=100, desc=f"Train epoch {epoch}"
        ):
            # get input (.to(self.device))
            targets = batch_data["targets"].to(self.device)
            targets_rating = batch_data["targets_rating"].to(
                self.device
            )  # B x max(target_length)
            targets_mask = batch_data["targets_mask"].to(self.device)
            negative_targets = batch_data["negative_targets"].to(self.device)
            negative_targets_mask = batch_data["negative_targets_mask"].to(self.device)
            user_embedding = batch_data["user_embedding"].to(self.device)
            user_embedding_mask = batch_data["user_embedding_mask"].to(self.device)
            targets_embedding = batch_data["targets_embedding"].to(self.device)
            targets_embedding_mask = batch_data["targets_embedding_mask"].to(
                self.device
            )
            negative_targets_embedding = batch_data["negative_targets_embedding"].to(
                self.device
            )
            negative_targets_embedding_mask = batch_data[
                "negative_targets_embedding_mask"
            ].to(self.device)

            # forward
            output = self.forward(
                targets,
                targets_rating,
                targets_mask,
                negative_targets,
                negative_targets_mask,
                user_embedding,
                user_embedding_mask,
                targets_embedding,
                targets_embedding_mask,
                negative_targets_embedding,
                negative_targets_embedding_mask,
            )

            # backward
            loss = output.loss
            loss.backward()
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()

            # log step
            log_info["step_losses"].append(loss.item())

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

        # loop
        for batch_data in tqdm.tqdm(
            dataloader, ncols=100, desc=f"Test {test_name}_data epoch {epoch}"
        ):
            # get input (.to(self.device))
            targets = batch_data["targets"].to(self.device)
            targets_rating = batch_data["targets_rating"].to(
                self.device
            )  # B x max(target_length)
            targets_mask = batch_data["targets_mask"].to(self.device)
            negative_targets = batch_data["negative_targets"].to(self.device)
            negative_targets_mask = batch_data["negative_targets_mask"].to(self.device)
            user_embedding = batch_data["user_embedding"].to(self.device)
            user_embedding_mask = batch_data["user_embedding_mask"].to(self.device)
            targets_embedding = batch_data["targets_embedding"].to(self.device)
            targets_embedding_mask = batch_data["targets_embedding_mask"].to(
                self.device
            )
            negative_targets_embedding = batch_data["negative_targets_embedding"].to(
                self.device
            )
            negative_targets_embedding_mask = batch_data[
                "negative_targets_embedding_mask"
            ].to(self.device)

            # forward
            with torch.no_grad():
                try:
                    output = self.forward(
                        targets,
                        targets_rating,
                        targets_mask,
                        negative_targets,
                        negative_targets_mask,
                        user_embedding,
                        user_embedding_mask,
                        targets_embedding,
                        targets_embedding_mask,
                        negative_targets_embedding,
                        negative_targets_embedding_mask,
                    )
                except:
                    # skip sample out of mem
                    print("skip sample out of mem:", negative_targets_embedding.shape)

            # log step
            step_losses.append(output.loss.item())

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
