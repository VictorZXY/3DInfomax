from typing import Dict, Callable

import torch

from trainer.trainer import Trainer


class PCBATrainer(Trainer):
    def __init__(self, model, args, metrics: Dict[str, Callable], main_metric: str, device: torch.device,
                 tensorboard_functions: Dict[str, Callable], optim=None, main_metric_goal: str = 'max',
                 loss_func=torch.nn.BCEWithLogitsLoss(), scheduler_step_per_batch: bool = True):
        super(PCBATrainer, self).__init__(model, args, metrics, main_metric, device, tensorboard_functions, optim,
                                          main_metric_goal, loss_func, scheduler_step_per_batch)

    def forward_pass(self, batch):
        targets = batch[-1]  # the last entry of the batch tuple is always the targets
        predictions = self.model(batch[0])  # foward the rest of the batch to the model
        is_labeled = (targets == targets)
        return self.loss_func(predictions[is_labeled], targets[is_labeled]), predictions, targets