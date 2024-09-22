import torch
from torchmetrics import Metric


class MAE(Metric):
    def __init__(self, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("steps", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("error", default=torch.tensor(0.0), dist_reduce_fx="sum")
    
    def update(self, p, y):
        self.steps += 1
        # Ensure p and y are both float tensors
        p = p.float()
        y = y.float()
        self.error += torch.mean(torch.abs(p - y))
    
    def compute(self):
        return self.error / self.steps
    

