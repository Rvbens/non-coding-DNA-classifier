import torch
import concurrent.futures as cf

#https://github.com/fastai/fastai/blob/master/fastai/metrics.py#L278
#faster and more precise than sklearn implementation

def roc_curve(input, targ):
    "Computes the receiver operator characteristic (ROC) curve by determining the true positive ratio (TPR) and false positive ratio (FPR) for various classification thresholds. Restricted binary classification tasks."
    targ = (targ == 1)
    desc_score_indices = input.argsort(descending=True)
    input = input.gather(-1,desc_score_indices)
    targ  = targ.gather(-1,desc_score_indices)
    d = input[1:] - input[:-1]
    distinct_value_indices = torch.nonzero(d).transpose(0,1)[0]
    threshold_idxs = torch.cat((distinct_value_indices, torch.LongTensor([len(targ) - 1]).to(targ.device)))
    tps = torch.cumsum(targ * 1, dim=-1)[threshold_idxs]
    fps = (1 + threshold_idxs - tps)
    if tps[0] != 0 or fps[0] != 0:
        zer = fps.new_zeros(1)
        fps = torch.cat((zer, fps))
        tps = torch.cat((zer, tps))
    fpr, tpr = fps.float() / fps[-1], tps.float() / tps[-1]
    return fpr, tpr

def auc_roc_score(input, targ):
    "Computes the area under the receiver operator characteristic (ROC) curve using the trapezoid method. Restricted binary classification tasks."
    fpr, tpr = roc_curve(input, targ)
    d = fpr[1:] - fpr[:-1]
    sl1, sl2 = [slice(None)], [slice(None)]
    sl1[-1], sl2[-1] = slice(1, None), slice(None, -1)
    return (d * (tpr[tuple(sl1)] + tpr[tuple(sl2)]) / 2.).sum(-1)

def weighted_auc(inputs,targets, weights):
    
    idxs = list(range(inputs.shape[1]))
    auc_i = lambda i: auc_roc_score(inputs[:,i],targets[:,i])
    
    with cf.ThreadPoolExecutor() as exc:
        results = exc.map(auc_i,idxs)
    results = list(results)

    weight_aucs = torch.stack(results)*weights
    weight_aucs[torch.isnan(weight_aucs)] = 0
    return weight_aucs.sum()

def accuracy(preds, targs):
    return ((preds>0.5)==targs).float().mean()
