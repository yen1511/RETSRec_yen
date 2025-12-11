import os
import requests
import json
import gzip


def process_data(url, output_dir):
    print("process ...", url)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # dowload data
    with open(os.path.join(output_dir, os.path.basename(url)), "wb") as f:
        r = requests.get(url, allow_redirects=True)
        f.write(r.content)

    # unzip
    with gzip.open(os.path.join(output_dir, os.path.basename(url)), "rb") as f_in:
        with open(
            os.path.join(output_dir, os.path.basename(url)).replace(".gz", ""), "wb"
        ) as f_out:
            f_out.write(f_in.read())

    # load data
    data = {}
    with open(
        os.path.join(output_dir, os.path.basename(url)).replace(".gz", ""), "r"
    ) as f:
        for l in f.readlines():
            review = eval(l)
            user_id = review["reviewerID"]
            if user_id not in data.keys():
                data[user_id] = []
            data[user_id].append(review)

    # xoá toàn bộ user có số item < 10
    for user_id in list(data.keys()):
        if len(data[user_id]) < 10:
            del data[user_id]

    # đếm số user
    print("# users", len(data.keys()))

    # đếm số item
    items = set()
    for user_id in data.keys():
        for review in data[user_id]:
            items.add(review["asin"])
    print("# items", len(items))

    # đếm số interaction
    interactions = 0
    for user_id in data.keys():
        interactions += len(data[user_id])
    print("# interactions", interactions)

    # sort data
    for user_id in data.keys():
        data[user_id].sort(key=lambda x: x["unixReviewTime"])

    # save data
    with open(os.path.join(output_dir, "data.json"), "w") as f:
        json.dump(data, f, indent=4)

    # split train/test: the first 70% of items in each user’s sequence as the training set and the rest are used for testing
    train_data = {}
    train_interactions = 0
    test_interactions = 0
    test_data = {}
    for user_id in data.keys():
        train_size = int(len(data[user_id]) * 0.7)
        train_data[user_id] = data[user_id][:train_size]
        test_data[user_id] = data[user_id][train_size:]

    with open(os.path.join(output_dir, "train.json"), "w") as f:
        json.dump(train_data, f, indent=4)

    with open(os.path.join(output_dir, "test.json"), "w") as f:
        json.dump(test_data, f, indent=4)

    for user_id in train_data.keys():
        train_interactions += len(train_data[user_id])
        test_interactions += len(test_data[user_id])
    print("# train interactions", train_interactions)
    print("# test interactions", test_interactions)


if __name__ == "__main__":
    process_data(
        url="https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Amazon_Instant_Video.json.gz",
        output_dir="./data/IV",
    )
    process_data(
        url="https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Pet_Supplies.json.gz",
        output_dir="./data/PS",
    )
    process_data(
        url="https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Tools_and_Home_Improvement.json.gz",
        output_dir="./data/THI",
    )
