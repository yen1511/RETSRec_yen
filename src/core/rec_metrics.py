import numpy as np


def _compute_apk(targets, predictions, k):
    if len(predictions) > k:
        predictions = predictions[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predictions):
        if p in targets and p not in predictions[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    if not list(targets):
        return 0.0
    return score / min(len(targets), k)


def _compute_precision_recall(targets, predictions, k):
    pred = predictions[:k]
    num_hit = len(set(pred).intersection(set(targets)))
    precision = float(num_hit) / len(pred)
    recall = float(num_hit) / len(targets)
    return precision, recall


def _compute_ndcg(targets, predictions, k):
    k1 = min(len(targets), k)
    if len(predictions) > k:
        predictions = predictions[:k]
    # compute idcg
    idcg = np.sum(1 / np.log2(np.arange(2, k1 + 2)))
    dcg = 0.0
    for i, p in enumerate(predictions):
        if p in targets:
            dcg += 1 / np.log2(i + 2)
    ndcg = dcg / idcg
    return ndcg


def _compute_hr(targets, predictions, k):
    pred = predictions[:k]
    for i in pred:
        if i in targets:
            return 1
    return 0
