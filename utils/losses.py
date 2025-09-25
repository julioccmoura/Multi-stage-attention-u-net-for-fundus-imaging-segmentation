import torch
import torch.nn as nn

# ------------- Losses -------------
def BCELoss():
    return nn.BCEWithLogitsLoss()


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super().__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth
    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        p, t = p.view(-1), targets.view(-1)
        TP = (p*t).sum(); FP = ((1-t)*p).sum(); FN = (t*(1-p)).sum()
        T = (TP + self.smooth) / (TP + self.alpha*FN + self.beta*FP + self.smooth)
        return 1 - T

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.smooth = alpha, beta, gamma, smooth
    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        p, t = p.view(-1), targets.view(-1)
        TP = (p*t).sum(); FP = ((1-t)*p).sum(); FN = (t*(1-p)).sum()
        T = (TP + self.smooth) / (TP + self.alpha*FN + self.beta*FP + self.smooth)
        return (1 - T).pow(self.gamma)
    
    
    