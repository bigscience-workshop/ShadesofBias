"""
File to compute metrics
"""
import pdb
from evaluate import evaluator, load

def accuracy_by_lang(data, target_lang, preds):
    """
    data: A HF dataset, with column of gold_lang and pred_lang
    returns: Eval result of HF accuracy
    """
    metric = load('accuracy', keep_in_memory=True)
    pdb.set_trace()
    return metric.compute(predictions=preds, references=data['gold'])

def accuracy_aggregate(data, preds):
    """
    data: A HF dataset, with column of gold_labellang and pred_labellang
    returns: Overall eval results
    """
    full_preds, golds = [], []
    metric = load('accuracy', keep_in_memory=True)
    for lang in preds:
        full_preds += preds[lang]
        golds += data['gold']
    pdb.set_trace()
    assert len(full_preds) == len(golds)
    return metric.compute(predictions=full_preds, references=golds)