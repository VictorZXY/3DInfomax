import copy
import os
from itertools import chain
from typing import Union, Tuple, Dict, Callable

import torch
from torch.utils.data import DataLoader

from commons.utils import move_to_device
from trainer.self_supervised_trainer import SelfSupervisedTrainer
from trainer.trainer import Trainer


class CLASSTrainer(Trainer):
    def __init__(self, model, model3d, critic, args, metrics: Dict[str, Callable], main_metric: str,
                 device: torch.device, tensorboard_functions: Dict[str, Callable],
                 optim=None, main_metric_goal: str = 'min', loss_func=torch.nn.MSELoss,
                 scheduler_step_per_batch: bool = True, **kwargs):
        self.model2 = model3d.to(device)  # move to device before loading optim params in super class
        self.critic = critic.to(device)
        super(CLASSTrainer, self).__init__(model, args, metrics, main_metric, device, tensorboard_functions,
                                                    optim, main_metric_goal, loss_func, scheduler_step_per_batch)

        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location=self.device)
            self.model2.load_state_dict(checkpoint['model2_state_dict'])


    def forward_pass(self, batch):
        graph = tuple(batch)[0]
        graph_copy = copy.deepcopy(graph)
        modelA_out = self.model(graph)  # foward the rest of the batch to the model
        modelB_out = self.model2(graph_copy)  # foward the rest of the batch to the model
        criticA_out = self.critic(modelA_out)
        criticB_out = self.critic(modelB_out)
        modelA_loss, modelB_loss, critic_loss, loss_components = self.loss_func(criticA_out, modelB_out, criticA_out, criticB_out)

        return modelA_loss, modelB_loss, critic_loss, (loss_components if loss_components != [] else None), modelA_out, modelB_out

    def process_batch(self, batch, optim):

        modelA_loss, modelB_loss, critic_loss, loss_components, predictions, targets = self.forward_pass(batch)
        if optim != None:  # run backpropagation if an optimizer is provided
            modelA_loss.backward(inputs=list(self.model.parameters()),
                                 retain_graph=True)
            self.optim.step()
            modelB_loss.backward(inputs=list(self.model2.parameters()),
                                 retain_graph=True)
            self.optim2.step()
            critic_loss.backward(inputs=list(self.critic.parameters()))
            self.optim_critic.step()

            self.optim.zero_grad()
            self.optim2.zero_grad()
            self.optim_critic.zero_grad()
            self.optim_steps += 1
        return modelA_loss, loss_components, (predictions.detach()), (targets.detach())

    def initialize_optimizer(self, optim):

        self.optim = optim(self.model.parameters(),**self.args.optimizer_params)
        self.optim2 = optim(self.model2.parameters(), **self.args.optimizer2_params)
        self.optim_critic = optim(self.critic.parameters(), **self.args.optimizer_critic_params)


    def save_model_state(self, epoch: int, checkpoint_name: str):
        torch.save({
            'epoch': epoch,
            'best_val_score': self.best_val_score,
            'optim_steps': self.optim_steps,
            'model_state_dict': self.model.state_dict(),
            'model2_state_dict': self.model2.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'scheduler_state_dict': None if self.lr_scheduler == None else self.lr_scheduler.state_dict()
        }, os.path.join(self.writer.log_dir, checkpoint_name))