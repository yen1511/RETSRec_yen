import os
import random
import json
import tqdm
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class AwsRecDataset(Dataset):
    def __init__(self, data_dir: str, data_type: str, sequence_length: int = 5):
        super().__init__()
        self.data_type = data_type
        self.sequence_length = sequence_length
        self.target_length = 1  # for train
        self.n_neg_samples = 3  # for train, for test 100 * target_length
        self.embedding_dim = 384
        if not os.path.isdir(data_dir):
            raise Exception()
        assert data_type in ["train", "test", "test_all"], (
            "data_type must be train or test"
        )
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
        if not os.path.isdir(self.item_embeddings_dir):
            self._create_item_embeddings(self.item_embeddings_dir)
        self.all_users = list(self.reviews["user"].keys())
        self.map_users = {id: idx for idx, id in enumerate(self.all_users)}
        self.all_items = list(self.reviews["item"].keys())
        self.map_items = {id: idx for idx, id in enumerate(self.all_items)}
        if self.data_type == "train":
            self.data = self._load_data_train()
        elif self.data_type in ["test", "test_all"]:
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

    def _create_rating_file(self, data_files, output_path: str):
        rating = {}
        for data_file in data_files:
            with open(data_file, "r") as f:
                data = json.load(f)
            for user_id, user_reviews in tqdm.tqdm(
                data.items(), desc=f"create_rating_file {data_file}"
            ):
                for review in user_reviews:
                    item_id = review["asin"]
                    rate_score = review["overall"]
                    rating[f"{user_id}_{item_id}"] = rate_score
        with open(output_path, "w") as f:
            json.dump(rating, f, indent=4)

    def _create_item_embeddings(self, output_dir):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        os.makedirs(output_dir, exist_ok=True)
        for item_id, reviews in tqdm.tqdm(self.reviews["item"].items()):
            reviews = [r["reviewText"] for r in reviews]
            embeddings = model.encode(reviews)
            np.save(os.path.join(output_dir, f"{item_id}.npy"), embeddings)

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
                user_items, window_size=self.sequence_length + self.target_length
            ):
                targets = sequences[self.sequence_length :]
                targets_rating = [self.rating[f"{user_id}_{i}"] for i in targets]
                sequences = sequences[: self.sequence_length]
                data.append(
                    {
                        "user_id": user_id,
                        "sequences": sequences,
                        "targets": targets,
                        "targets_rating": targets_rating,
                    }
                )
        return data

    def _load_data_test(self):
        data = []
        for user_id, user_items in self.test.items():
            targets = [i["asin"] for i in user_items]
            targets_rating = [self.rating[f"{user_id}_{i}"] for i in targets]
            sequences = [i["asin"] for i in self.train[user_id]][
                -self.sequence_length :
            ]
            data.append(
                {
                    "user_id": user_id,
                    "sequences": sequences,
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
        elif self.data_type == "test_all":
            n_neg_samples = len(self.all_items)
        else:
            raise Exception()
        sample["negative_targets"] = self._generate_negative_targets(
            sample["user_id"], n_neg_samples
        )
        return sample

    def collate_fn(self, batch_data):
        B = len(batch_data)

        # get id tensor
        # batch_user_id --> B
        # batch_sequences --> B x sequence_length
        # batch_targets --> B x target_length (for train target_length is same, for test we need padding)
        # batch_negative_targets --> B x n_neg_samples (for train n_neg_samples is same, for test we need padding)
        batch_user_id = torch.tensor(
            [self.map_users[sample["user_id"]] for sample in batch_data]
        )
        batch_sequences = torch.tensor(
            [
                [self.map_items[item_id] + 1 for item_id in sample["sequences"]]
                for sample in batch_data
            ]
        )
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
        elif self.data_type in ["test", "test_all"]:
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
        elif self.data_type in ["test", "test_all"]:
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

        # get embedding tensor
        # batch_sequences_text_embedding --> B x sequence_length x n_review x embedding_dim (n_review need padding)
        # batch_targets_text_embedding --> B x target_length x n_review x embedding_dim (for train target_length is same, for test we need padding, n_review need padding)
        # batch_negative_targets_text_embedding --> B x n_neg_samples x n_review x embedding_dim (for train n_neg_samples is same, for test we need padding, n_review need padding)
        batch_sequences_text_embedding = []
        batch_targets_text_embedding = []
        batch_negative_targets_text_embedding = []
        for sample in batch_data:
            sequences_text_embedding = []
            for item_id in sample["sequences"]:
                item_embedding = np.load(
                    os.path.join(self.item_embeddings_dir, f"{item_id}.npy")
                )
                sequences_text_embedding.append(item_embedding)
            batch_sequences_text_embedding.append(sequences_text_embedding)
            targets_text_embedding = []
            for item_id in sample["targets"]:
                item_embedding = np.load(
                    os.path.join(self.item_embeddings_dir, f"{item_id}.npy")
                )
                targets_text_embedding.append(item_embedding)
            batch_targets_text_embedding.append(targets_text_embedding)
            negative_targets_text_embedding = []
            for item_id in sample["negative_targets"]:
                item_embedding = np.load(
                    os.path.join(self.item_embeddings_dir, f"{item_id}.npy")
                )
                negative_targets_text_embedding.append(item_embedding)
            batch_negative_targets_text_embedding.append(
                negative_targets_text_embedding
            )
        max_n_review_sequences = max(
            [len(item) for sample in batch_sequences_text_embedding for item in sample]
        )
        batch_sequences_text_embedding_tensor = torch.tensor(
            np.array(
                [
                    [
                        np.pad(item, ((0, max_n_review_sequences - len(item)), (0, 0)))
                        for item in sample
                    ]
                    for sample in batch_sequences_text_embedding
                ]
            ),
            dtype=torch.float32,
        )
        if self.data_type == "train":
            max_n_review_targets = max(
                [
                    len(item)
                    for sample in batch_targets_text_embedding
                    for item in sample
                ]
            )
            batch_targets_text_embedding_tensor = torch.tensor(
                np.array(
                    [
                        [
                            np.pad(
                                item, ((0, max_n_review_targets - len(item)), (0, 0))
                            )
                            for item in sample
                        ]
                        for sample in batch_targets_text_embedding
                    ]
                ),
                dtype=torch.float32,
            )
            max_n_review_negative_targets = max(
                [
                    len(item)
                    for sample in batch_negative_targets_text_embedding
                    for item in sample
                ]
            )
            batch_negative_targets_text_embedding_tensor = torch.tensor(
                np.array(
                    [
                        [
                            np.pad(
                                item,
                                (
                                    (0, max_n_review_negative_targets - len(item)),
                                    (0, 0),
                                ),
                            )
                            for item in sample
                        ]
                        for sample in batch_negative_targets_text_embedding
                    ]
                ),
                dtype=torch.float32,
            )
        elif self.data_type in ["test", "test_all"]:
            max_n_targets = max(
                [len(sample) for sample in batch_targets_text_embedding]
            )
            max_n_review_targets = max(
                [
                    len(item)
                    for sample in batch_targets_text_embedding
                    for item in sample
                ]
            )
            batch_targets_text_embedding_tensor = torch.tensor(
                np.array(
                    [
                        [
                            np.pad(
                                item, ((0, max_n_review_targets - len(item)), (0, 0))
                            )
                            for item in sample
                        ]
                        + [np.zeros((max_n_review_targets, self.embedding_dim))]
                        * (max_n_targets - len(sample))
                        for sample in batch_targets_text_embedding
                    ]
                ),
                dtype=torch.float32,
            )
            max_n_negative_targets = max(
                [len(sample) for sample in batch_negative_targets_text_embedding]
            )
            max_n_review_negative_targets = max(
                [
                    len(item)
                    for sample in batch_negative_targets_text_embedding
                    for item in sample
                ]
            )
            batch_negative_targets_text_embedding_tensor = torch.tensor(
                np.array(
                    [
                        [
                            np.pad(
                                item,
                                (
                                    (0, max_n_review_negative_targets - len(item)),
                                    (0, 0),
                                ),
                            )
                            for item in sample
                        ]
                        + [
                            np.zeros(
                                (max_n_review_negative_targets, self.embedding_dim)
                            )
                        ]
                        * (max_n_negative_targets - len(sample))
                        for sample in batch_negative_targets_text_embedding
                    ]
                ),
                dtype=torch.float32,
            )
        else:
            raise Exception()

        # batch_sequences_text_embedding_mask --> B x sequence_length x max(n_review)
        batch_sequences_text_embedding_mask = torch.ones(
            (B, self.sequence_length, max_n_review_sequences), dtype=torch.int32
        )
        for i in range(B):
            for j in range(self.sequence_length):
                batch_sequences_text_embedding_mask[
                    i, j, len(batch_sequences_text_embedding[i][j]) :
                ] = 0

        # batch_targets_text_embedding_mask --> B x max(target_length) x max(n_review)
        batch_targets_text_embedding_mask = torch.ones(
            (B, batch_targets.shape[1], max_n_review_targets), dtype=torch.int32
        )
        for i in range(B):
            for j in range(batch_targets.shape[1]):
                if j >= len(batch_targets_text_embedding[i]):
                    batch_targets_text_embedding_mask[i, j, :] = 0
                else:
                    batch_targets_text_embedding_mask[
                        i, j, len(batch_targets_text_embedding[i][j]) :
                    ] = 0

        # batch_negative_targets_text_embedding_mask --> B x max(n_neg_samples) x max(n_review)
        batch_negative_targets_text_embedding_mask = torch.ones(
            (B, batch_negative_targets.shape[1], max_n_review_negative_targets),
            dtype=torch.int32,
        )
        for i in range(B):
            for j in range(batch_negative_targets.shape[1]):
                if j >= len(batch_negative_targets_text_embedding[i]):
                    batch_negative_targets_text_embedding_mask[i, j, :] = 1
                else:
                    batch_negative_targets_text_embedding_mask[
                        i, j, len(batch_negative_targets_text_embedding[i][j]) :
                    ] = 0

        return {
            "user_id": batch_user_id,  # B
            "sequences": batch_sequences,  # B x sequence_length
            "targets": batch_targets,  # B x max(target_length)
            "targets_rating": batch_targets_rating,  # B x max(target_length)
            "targets_mask": batch_targets_mask,  # B x max(target_length)
            "negative_targets": batch_negative_targets,  # B x max(n_neg_samples)
            "negative_targets_mask": batch_negative_targets_mask,  # B x max(n_neg_samples)
            "sequences_text_embedding": batch_sequences_text_embedding_tensor,  # B x sequence_length x max(n_review) x embedding_dim
            "sequences_text_embedding_mask": batch_sequences_text_embedding_mask,  # B x sequence_length x max(n_review)
            "targets_text_embedding": batch_targets_text_embedding_tensor,  # B x max(target_length) x max(n_review) x embedding_dim
            "targets_text_embedding_mask": batch_targets_text_embedding_mask,  # B x max(target_length) x max(n_review)
            "negative_targets_text_embedding": batch_negative_targets_text_embedding_tensor,  # B x max(n_neg_samples) x max(n_review) x embedding_dim
            "negative_targets_text_embedding_mask": batch_negative_targets_text_embedding_mask,  # B x max(n_neg_samples) x max(n_review)
        }


if __name__ == "__main__":
    import tqdm

    data_train = AwsRecDataset("./data/IV", "test")
    dataloader_train = DataLoader(
        data_train,
        batch_size=1,
        shuffle=True,
        num_workers=10,
        collate_fn=data_train.collate_fn,
    )
    for i in tqdm.tqdm(dataloader_train):
        pass
        # print(i["user_id"].shape)
        # print(i["sequences"].shape)
        # print(i["targets"].shape)
        # print(i["targets_rating"].shape)
        # print(i["targets_rating"])
        # print(i["targets_mask"].shape)
        # print(i["negative_targets"].shape)
        # print(i["negative_targets_mask"].shape)
        # print(i["sequences_text_embedding"].shape)
        # print(i["sequences_text_embedding_mask"].shape)
        # print(i["targets_text_embedding"].shape)
        # print(i["targets_text_embedding_mask"].shape)
        # print(i["negative_targets_text_embedding"].shape)
        # print(i["negative_targets_text_embedding_mask"].shape)
        # exit()
