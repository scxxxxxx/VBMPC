import torch


class Upsample(torch.nn.Module):
    def __init__(self,  size=None, mode='bilinear', align_corners=True):
        super(Upsample, self).__init__()
        self.size = (480, 640) if size is None else size
        self.mode = mode
        self.align_corners = align_corners
    def forward(self, x):
        return torch.nn.functional.interpolate(x, size=self.size, mode=self.mode, align_corners=self.align_corners)


class SelectItem(torch.nn.Module):
    """
    this is used in the nn.Sequential to select the item from a tuple, which is the output from a previous module.
    """
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]


class SelectRNNItem(torch.nn.Module):
    """
    this is used in the nn.Sequential to select the item from a tuple, which is the output from a previous module.
    (b,T,dir∗d)
    last step only: if bidirectional case, (b, dir * d), else: (b, d)
    all steps: (b, T, dir*d)
    """
    def __init__(self, output_only: bool = True, last_step_only: bool = True, bidirectional: bool = False,
                 merge_mode: str = 'sum'):
        super(SelectRNNItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = 0 if output_only else 1
        self.last_step_only = last_step_only
        self.bidirectional = bidirectional
        self.merge_mode = merge_mode

    def forward(self, inputs):
        """
        (b,T,dir∗d)
        """
        out = inputs[self.item_index]
        if self.last_step_only:
            if self.bidirectional:
                forward, reverse = torch.split(out, torch.div(out.shape[-1], 2, rounding_mode='floor'), dim=-1)
                # forward, reverse = torch.split(out, out.shape[-1] // 2, dim=-1)
                out = torch.cat((forward[-1], reverse[0]), dim=-1)
            else:
                out = out[:, -1, :]
        else:
            if self.bidirectional:
                forward, reverse = torch.split(out, torch.div(out.shape[-1], 2, rounding_mode='floor'), dim=-1)
                # forward, reverse = torch.split(out, out.shape[-1] // 2, dim=-1)
                if self.merge_mode == "sum":
                    out = forward + torch.flip(reverse, dims=[1])
                if self.merge_mode == "cat":
                    out = torch.cat((forward, torch.flip(reverse, dims=[1])), dim=-1)
        return out


class Transpose(torch.nn.Module):
    """
    this is used in the nn.Sequential to select the item from a tuple, which is the output from a previous module.
    """
    def __init__(self, dim0: int, dim1: int):
        super(Transpose, self).__init__()
        self._name = 'Transpose'
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, inputs: torch.tensor):
        return inputs.transpose(self.dim0, self.dim1)
