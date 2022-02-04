from itertools import chain
from typing import Union, Tuple, Dict, Callable

import torch
from torch.utils.data import DataLoader

from commons.utils import move_to_device
from trainer.self_supervised_trainer import SelfSupervisedTrainer
from trainer.trainer import Trainer


class CLASSTrainer(Trainer):
    def __init__(self, model, model3d, args, metrics: Dict[str, Callable], main_metric: str,
                 device: torch.device, tensorboard_functions: Dict[str, Callable],
                 optim=None, main_metric_goal: str = 'min', loss_func=torch.nn.MSELoss,
                 scheduler_step_per_batch: bool = True, **kwargs):
        self.model3d = model3d.to(device)  # move to device before loading optim params in super class
        super(SelfSupervisedTrainer, self).__init__(model, args, metrics, main_metric, device, tensorboard_functions,
                                                    optim, main_metric_goal, loss_func, scheduler_step_per_batch)

        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location=self.device)
            self.model3d.load_state_dict(checkpoint['model3d_state_dict'])

    def forward_pass(self, batch):
        graph, info3d, distances = tuple(batch)
        view2d = self.model(*graph)  # foward the rest of the batch to the model
        view3d, distance_preds = self.model3d(*info3d)
        loss_contrastive, loss_reconstruction = self.loss_func(view2d, view3d, distance_preds, distances)
        return loss_contrastive, loss_reconstruction, view2d, view3d

    def process_batch(self, batch, optim):
        loss_contrastive,loss_reconstruction, predictions, targets = self.forward_pass(batch)
        loss = loss_contrastive + loss_reconstruction
        if optim != None:  # run backpropagation if an optimizer is provided
            loss.backward()
            self.optim.step()
            self.after_optim_step()  # overwrite this function to do stuff before zeroing out grads
            self.optim.zero_grad()
            self.optim_steps += 1
        return loss_contrastive, loss_reconstruction, predictions.detach(), targets.detach()

    def initialize_optimizer(self, optim):
        normal_params = [v for k, v in chain(self.model.named_parameters(), self.model3d.named_parameters()) if
                         not 'batch_norm' in k]
        batch_norm_params = [v for k, v in chain(self.model.named_parameters(), self.model3d.named_parameters()) if
                             'batch_norm' in k]

        self.optim = optim([{'params': batch_norm_params, 'weight_decay': 0},
                            {'params': normal_params}],
                           **self.args.optimizer_params)

    def save_model_state(self, epoch: int, checkpoint_name: str):
        torch.save({
            'epoch': epoch,
            'best_val_score': self.best_val_score,
            'optim_steps': self.optim_steps,
            'model_state_dict': self.model.state_dict(),
            'model3d_state_dict': self.model3d.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'scheduler_state_dict': None if self.lr_scheduler == None else self.lr_scheduler.state_dict()
        }, os.path.join(self.writer.log_dir, checkpoint_name))