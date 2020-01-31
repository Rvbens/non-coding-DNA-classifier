import torch
import concurrent.futures as cf

#https://github.com/fastai/fastai/blob/master/fastai/metrics.py#L278
#faster and more precise than sklearn implementation

def roc_pr_curves(input, targ):
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
    ppv = tps.float()/(tps+fps).float()
    ppv[0]= 0 #0/0 = nan
    return fpr, tpr, ppv

def auc_scores(input, targ):
    fpr, tpr, ppv = roc_pr_curves(input, targ)
    d = fpr[1:] - fpr[:-1]
    sl1, sl2 = [slice(None)], [slice(None)]
    sl1[-1], sl2[-1] = slice(1, None), slice(None, -1)
    roc_auc = (d * (tpr[tuple(sl1)] + tpr[tuple(sl2)]) / 2.).sum(-1)
    #plt.plot(fpr.numpy(),tpr.numpy())
    
    d = tpr[1:] - tpr[:-1]
    sl1, sl2 = [slice(None)], [slice(None)]
    sl1[-1], sl2[-1] = slice(1, None), slice(None, -1)
    pr_auc = (d * (ppv[tuple(sl1)] + ppv[tuple(sl2)]) / 2.).sum(-1)
    #plt.plot(tpr.numpy(),ppv.numpy())
    return roc_auc,pr_auc


def weighted_aucs(inputs,targets, weights):
    idxs = list(range(inputs.shape[1]))
    auc_i = lambda i: auc_scores(inputs[:,i],targets[:,i])

    with cf.ThreadPoolExecutor() as exc:
        results = exc.map(auc_i,idxs)
    results = torch.stack([torch.stack(i) for i in results])
    
    weight_aucs = results*weights[:,None]
    weight_aucs[torch.isnan(weight_aucs)] = 0
    roc_auc, pr_auc =  weight_aucs.sum(0)
    return roc_auc, pr_auc

def accuracy(preds, targs):
    return ((preds>0.5)==targs).float().mean()
