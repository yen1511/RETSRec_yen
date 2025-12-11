import os
import random
import json
import tqdm
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class AwsRecOldDataset(Dataset):
    def __init__(self, data_dir: str, data_type: str):
        super().__init__()
        self.data_type = data_type
        self.target_length = 1  # for train
        self.n_neg_samples = 3  # for train, for test 100 * target_length
        self.embedding_dim = 384
        if not os.path.isdir(data_dir):
            raise Exception()
        assert data_type in ["train", "test"], "data_type must be train or test"
        assert os.path.isfile(os.path.join(data_dir, "train.json")), (
            "train.json is not found"
        )
        assert os.path.isfile(os.path.join(data_dir, "test.json")), (
            "train.json is not found"
        )
        if not os.path.isfile(os.path.join(data_dir, "reviews.json")):
            self._create_reviews_file(
                data_files=[
                    os.path.join(data_dir, "train.json"),
                    os.path.join(data_dir, "test.json"),
                ],
                output_path=os.path.join(data_dir, "reviews.json"),
            )
        if not os.path.isfile(os.path.join(data_dir, "rating.json")):
            self._create_rating_file(
                data_files=[
                    os.path.join(data_dir, "train.json"),
                    os.path.join(data_dir, "test.json"),
                ],
                output_path=os.path.join(data_dir, "rating.json"),
            )
        with open(os.path.join(data_dir, "train.json"), "r") as f:
            self.train = json.load(f)
        with open(os.path.join(data_dir, "test.json"), "r") as f:
            self.test = json.load(f)
        with open(os.path.join(data_dir, "reviews.json"), "r") as f:
            self.reviews = json.load(f)
        with open(os.path.join(data_dir, "rating.json"), "r") as f:
            self.rating = json.load(f)
        self.item_embeddings_dir = os.path.join(data_dir, "item_embeddings")
        self.user_embeddings_dir = os.path.join(data_dir, "user_embeddings")
        if not os.path.isdir(self.item_embeddings_dir):
            self._create_item_embeddings(self.item_embeddings_dir)
        if not os.path.isdir(self.user_embeddings_dir):
            self._create_user_embeddings(self.user_embeddings_dir)
        self.all_users = list(self.reviews["user"].keys())
        self.map_users = {id: idx for idx, id in enumerate(self.all_users)}
        self.all_items = list(self.reviews["item"].keys())
        self.map_items = {id: idx for idx, id in enumerate(self.all_items)}
        if self.data_type == "train":
            self.data = self._load_data_train()
        elif self.data_type == "test":
            self.data = self._load_data_test()
        else:
            raise Exception()

    def _create_reviews_file(self, data_files, output_path: str):
        reviews = {"user": {}, "item": {}}
        for data_file in data_files:
            with open(data_file, "r") as f:
                data = json.load(f)
            for user_id, user_reviews in tqdm.tqdm(
                data.items(), desc=f"create_reviews_file {data_file}"
            ):
                if user_id not in reviews["user"].keys():
                    reviews["user"][user_id] = []
                reviews["user"][user_id].extend(user_reviews)
                for review in user_reviews:
                    item_id = review["asin"]
                    if item_id not in reviews["item"].keys():
                        reviews["item"][item_id] = []
                    reviews["item"][item_id].append(review)
        for user_id in reviews["user"].keys():
            reviews["user"][user_id].sort(key=lambda x: x["unixReviewTime"])
        for item_id in reviews["item"].keys():
            reviews["item"][item_id].sort(key=lambda x: x["unixReviewTime"])
        with open(output_path, "w") as f:
            json.dump(reviews, f, indent=4)

    def _create_item_embeddings(self, output_dir):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        os.makedirs(output_dir, exist_ok=True)
        for item_id, reviews in tqdm.tqdm(self.reviews["item"].items()):
            reviews = [r["reviewText"] for r in reviews]
            embeddings = model.encode(reviews)
            np.save(os.path.join(output_dir, f"{item_id}.npy"), embeddings)

    def _create_user_embeddings(self, output_dir):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        os.makedirs(output_dir, exist_ok=True)
        for user_id, reviews in tqdm.tqdm(self.reviews["user"].items()):
            test_item_ids = [r["asin"] for r in self.test[user_id]]
            reviews = [
                r["reviewText"] for r in reviews if r["asin"] not in test_item_ids
            ]
            embeddings = model.encode(reviews)
            np.save(os.path.join(output_dir, f"{user_id}.npy"), embeddings)

    def _sliding_window(self, inputs, window_size, step_size=1):
        for start_idx in range(0, len(inputs) - window_size + 1, step_size):
            end_idx = start_idx + window_size
            yield inputs[start_idx:end_idx]

    def _generate_negative_targets(self, user_id, n_neg_samples):
        candidates = list(
            set(self.all_items)
            - set([i["asin"] for i in self.reviews["user"][user_id]])
        )
        if n_neg_samples < len(candidates):
            negative_targets = random.sample(candidates, n_neg_samples)
            return negative_targets
        else:
            return candidates

    def _load_data_train(self):
        data = []
        for user_id, user_items in self.train.items():
            user_items = [i["asin"] for i in user_items]
            for sequences in self._sliding_window(
                user_items, window_size=self.target_length
            ):
                targets_rating = [self.rating[f"{user_id}_{i}"] for i in sequences]
                data.append(
                    {
                        "user_id": user_id,
                        "targets": sequences,
                        "targets_rating": targets_rating,
                    }
                )
        return data

    def _load_data_test(self):
        data = []
        for user_id, user_items in self.test.items():
            targets = [i["asin"] for i in user_items]
            targets_rating = [self.rating[f"{user_id}_{i}"] for i in targets]
            data.append(
                {
                    "user_id": user_id,
                    "targets": targets,
                    "targets_rating": targets_rating,
                }
            )
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        if self.data_type == "train":
            n_neg_samples = self.n_neg_samples
        elif self.data_type == "test":
            n_neg_samples = 100 * len(sample["targets"])
        else:
            raise Exception()
        sample["negative_targets"] = self._generate_negative_targets(
            sample["user_id"], n_neg_samples
        )
        return sample

    def _get_batch_text_embedding(self, batch_data, embeddings_dir):
        B = len(batch_data)

        # get list sample (each sample has list item; each item has list emb)
        # batch --> samples --> items --> reviews
        batch_text_embedding = []
        for sample in batch_data:
            text_embedding = []
            for _id in sample:
                _embedding = np.load(os.path.join(embeddings_dir, f"{_id}.npy"))
                text_embedding.append(_embedding)
            batch_text_embedding.append(text_embedding)

        # get max n item & n review
        max_n_item = max([len(sample) for sample in batch_text_embedding])
        max_n_review = max([len(i) for sample in batch_text_embedding for i in sample])

        # get tensor --> B x max(n_item) x max(n_review)
        batch_text_embedding_tensor = torch.tensor(
            np.array(
                [
                    [
                        np.pad(item, ((0, max_n_review - len(item)), (0, 0)))
                        for item in sample
                    ]
                    + [np.zeros((max_n_review, self.embedding_dim))]
                    * (max_n_item - len(sample))
                    for sample in batch_text_embedding
                ]
            ),
            dtype=torch.float32,
        )

        # batch_targets_text_embedding_mask --> B x max(target_length) x max(n_review)
        batch_targets_text_embedding_mask = torch.ones(
            (B, max_n_item, max_n_review), dtype=torch.int32
        )
        for sample_idx, sample in enumerate(batch_text_embedding):
            batch_targets_text_embedding_mask[sample_idx, len(sample) :, :] = 0
            for item_idx, item in enumerate(sample):
                batch_targets_text_embedding_mask[
                    sample_idx, item_idx:, item.shape[0] :
                ] = 0

        return batch_text_embedding_tensor, batch_targets_text_embedding_mask

    def collate_fn(self, batch_data):
        if self.data_type == "train":
            batch_targets = torch.tensor(
                [
                    [self.map_items[item_id] + 1 for item_id in sample["targets"]]
                    for sample in batch_data
                ]
            )
            batch_negative_targets = torch.tensor(
                [
                    [
                        self.map_items[item_id] + 1
                        for item_id in sample["negative_targets"]
                    ]
                    for sample in batch_data
                ]
            )
        elif self.data_type == "test":
            max_n_targets = max([len(sample["targets"]) for sample in batch_data])
            batch_targets = torch.tensor(
                [
                    [self.map_items[item_id] + 1 for item_id in sample["targets"]]
                    + [0] * (max_n_targets - len(sample["targets"]))
                    for sample in batch_data
                ]
            )
            max_n_negative_targets = max(
                [len(sample["negative_targets"]) for sample in batch_data]
            )
            batch_negative_targets = torch.tensor(
                [
                    [
                        self.map_items[item_id] + 1
                        for item_id in sample["negative_targets"]
                    ]
                    + [0] * (max_n_negative_targets - len(sample["negative_targets"]))
                    for sample in batch_data
                ]
            )
        else:
            raise Exception()

        # get rating tensor
        if self.data_type == "train":
            batch_targets_rating = torch.tensor(
                [sample["targets_rating"] for sample in batch_data]
            )
        elif self.data_type == "test":
            max_n_targets = max(
                [len(sample["targets_rating"]) for sample in batch_data]
            )
            batch_targets_rating = torch.tensor(
                [
                    sample["targets_rating"]
                    + [0] * (max_n_targets - len(sample["targets_rating"]))
                    for sample in batch_data
                ]
            )
        else:
            raise Exception()

        # batch_targets_mask --> B x target_length
        batch_targets_mask = torch.ones_like(batch_targets)
        batch_targets_mask[batch_targets == 0] = 0

        # batch_negative_targets_mask --> B x n_neg_samples
        batch_negative_targets_mask = torch.ones_like(batch_negative_targets)
        batch_negative_targets_mask[batch_negative_targets == 0] = 0

        user_embedding, user_embedding_mask = self._get_batch_text_embedding(
            [[sample["user_id"]] for sample in batch_data], self.user_embeddings_dir
        )
        targets_embedding, targets_embedding_mask = self._get_batch_text_embedding(
            [sample["targets"] for sample in batch_data], self.item_embeddings_dir
        )
        negative_targets_embedding, negative_targets_embedding_mask = (
            self._get_batch_text_embedding(
                [sample["negative_targets"] for sample in batch_data],
                self.item_embeddings_dir,
            )
        )

        return {
            "targets": batch_targets,  # B x max(target_length)
            "targets_rating": batch_targets_rating,  # B x max(target_length)
            "targets_mask": batch_targets_mask,  # B x max(target_length)
            "negative_targets": batch_negative_targets,  # B x max(n_neg_samples)
            "negative_targets_mask": batch_negative_targets_mask,  # B x max(n_neg_samples)
            "user_embedding": user_embedding,
            "user_embedding_mask": user_embedding_mask,
            "targets_embedding": targets_embedding,
            "targets_embedding_mask": targets_embedding_mask,
            "negative_targets_embedding": negative_targets_embedding,
            "negative_targets_embedding_mask": negative_targets_embedding_mask,
        }


if __name__ == "__main__":
    import tqdm

    data_train = AwsRecOldDataset("./data/IV", "test")
    dataloader_train = DataLoader(
        data_train,
        batch_size=8,
        shuffle=True,
        num_workers=1,
        collate_fn=data_train.collate_fn,
    )
    for i in tqdm.tqdm(dataloader_train):
        pass
