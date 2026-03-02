import torch
from typing import Optional, Tuple


class RNNCache:
    def __init__(self):
        self.enabled = False
        self.hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def enable(self):
        self.enabled = True
        self.hidden = None

    def disable(self):
        self.enabled = False
        self.hidden = None

    def reset(self):
        self.hidden = None

    def get(
        self, batch_size: int, lengths
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if not self.enabled:
            return None
        if batch_size != 1 or lengths is not None:
            return None
        return self.hidden

    def store(self, hidden, cell):
        if self.enabled:
            self.hidden = (hidden.detach(), cell.detach())
