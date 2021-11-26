import torch


class iLQR(torch.nn.Module):
    def __init__(self):
        super(iLQR, self).__init__()

    def forward(self, s, a):
        raise NotImplementedError
