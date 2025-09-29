import torch
import torch.nn.functional as F
def info_nce_loss(query,positive,negatives,temperature=0.1):
    pos_sim=torch.matmul(query,positive.T)/temperature
    neg_sim=torch.matmul(query,negatives.tanspose(0,2,1))/temperature

    logits=torch.cat([pos_sim.unsqueeze(1),neg_sim],dim=1)
    labels=torch.zeros(logits.size(0),dtype=torch.long)

    return F.cross_entropy(logits,labels)
