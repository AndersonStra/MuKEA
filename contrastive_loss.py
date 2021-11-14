import torch
from torch import nn


def distanceL2(h, t):
    s = h - t
    sum = torch.square(s).sum(-1)
    return sum

def dot_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())

def l2_sim(im, s):
    b_im = im.shape[0]
    b_s = s.shape[0]
    return distanceL2(im.unsqueeze(0).repeat(b_s,1,1),s.unsqueeze(1).repeat(1,b_im,1)).transpose(0,1)


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=1.0, measure=False, max_violation=False):
        # max_violation 是否用最难样本
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        if measure == 'l2':
            self.sim = l2_sim
        if measure == 'dot':
            self.sim = dot_sim

        self.max_violation = max_violation

    def forward(self, im, s):
        # compute image-sentence score matrix
        #im,s维度相同，默认将除了配对的都视为负样本
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # # clear diagonals
        # mask = torch.eye(scores.size(0)) > .5
        # I = mask
        # if torch.cuda.is_available():
        #     I = I.cuda()
        # cost_s = cost_s.masked_fill_(I, 0)
        # cost_im = cost_im.masked_fill_(I, 0)

        # another mask method
        mask1 = scores.eq(d1).cuda()
        mask2 = mask1.t()
        cost_s = cost_s.masked_fill_(mask1, 0)
        cost_im = cost_im.masked_fill_(mask2, 0)



        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()
