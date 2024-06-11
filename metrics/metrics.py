import math


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
