"""
File to compute metrics
"""

import math

from evaluate import evaluator, load


def accuracy_by_lang(data, preds):
    """
    data: A HF dataset, with column of gold_lang and pred_lang
    returns: Eval result of HF accuracy
    """
    metric = load("accuracy", keep_in_memory=True)
    return metric.compute(predictions=preds, references=data["gold"])


def accuracy_aggregate(data, preds):
    """
    data: A HF dataset, with column of gold_labellang and pred_labellang
    returns: Overall eval results
    """
    full_preds, golds = [], []
    metric = load("accuracy", keep_in_memory=True)
    for lang in preds:
        full_preds += preds[lang]
        golds += data["gold"]
    assert len(full_preds) == len(golds)
    return metric.compute(predictions=full_preds, references=golds)


def mean(arr):
    return sum(arr) / len(arr)


def perplexity(items):
    return math.exp(-mean(items))


def neg_log_likelihood(items):
    neg_log_likelihood = -mean(items)
    return neg_log_likelihood


def weighted_mean(items):
    a, b = zip(*items)
    return sum(a) / sum(b)


def weighted_perplexity(items):
    return math.exp(-weighted_mean(items))
