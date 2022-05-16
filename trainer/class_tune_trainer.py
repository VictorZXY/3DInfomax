import copy
import os
import shutil
from typing import Dict, Callable

import torch
from torch.utils.data import DataLoader

from commons.utils import tensorboard_singular_value_plot
from trainer.trainer import Trainer


class CLASSFrozenFinetuneTrainer(Trainer):
    def __init__(self, model, critic, args, metrics: Dict[str, Callable], main_metric: str,
                 device: torch.device, tensorboard_functions: Dict[str, Callable],
                 optim=None, main_metric_goal: str = 'min', loss_func=torch.nn.L1Loss,
                 scheduler_step_per_batch: bool = True, **kwargs):
        # move to device before loading optim params in super class
        # no need to move model because it will be moved in super class call
        self.critic = critic.to(device)
        super(CLASSFrozenFinetuneTrainer, self).__init__(model, args, metrics, main_metric, device,
                                                         tensorboard_functions, optim, main_metric_goal, loss_func,
                                                         scheduler_step_per_batch)

        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location=self.device)
            self.critic.load_state_dict(checkpoint['critic_state_dict'])

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        epochs_no_improve = 0  # counts every epoch that the validation accuracy did not improve for early stopping
        for epoch in range(self.start_epoch, self.args.num_epochs + 1):  # loop over the dataset multiple times
            self.model.train()
            self.predict(train_loader, epoch, optim=self.optim)

            self.model.eval()
            with torch.no_grad():
                metrics = self.predict(val_loader, epoch)
                val_score = metrics[self.main_metric]

                if self.lr_scheduler != None and not self.scheduler_step_per_batch:
                    self.step_schedulers(metrics=val_score)

                if self.args.eval_per_epochs > 0 and epoch % self.args.eval_per_epochs == 0:
                    self.run_per_epoch_evaluations([(train_loader, 'train'), (val_loader, 'val')])

                self.tensorboard_log(metrics, data_split='val', epoch=epoch, log_hparam=True, step=self.optim_steps)
                val_loss = metrics[type(self.loss_func).__name__]
                print('[Epoch %d] %s: %.6f val loss: %.6f' % (epoch, self.main_metric, val_score, val_loss))

                # save the model with the best main_metric depending on wether we want to maximize or minimize the main metric
                if val_score >= self.best_val_score and self.main_metric_goal == 'max' or val_score <= self.best_val_score and self.main_metric_goal == 'min':
                    epochs_no_improve = 0
                    self.best_val_score = val_score
                    self.save_checkpoint(epoch, checkpoint_name='best_checkpoint.pt')
                else:
                    epochs_no_improve += 1
                self.save_checkpoint(epoch, checkpoint_name='last_checkpoint.pt')

                if epochs_no_improve >= self.args.patience and epoch >= self.args.minimum_epochs:  # stopping criterion
                    print(
                        f'Early stopping criterion based on -{self.main_metric}- that should be {self.main_metric_goal} reached after {epoch} epochs. Best model checkpoint was in epoch {epoch - epochs_no_improve}.')
                    break
                if epoch in self.args.models_to_save:
                    shutil.copyfile(os.path.join(self.writer.log_dir, 'best_checkpoint.pt'),
                                    os.path.join(self.writer.log_dir, f'best_checkpoint_{epoch}epochs.pt'))

        # evaluate on best checkpoint
        checkpoint = torch.load(os.path.join(self.writer.log_dir, 'best_checkpoint.pt'), map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return self.evaluation(val_loader, data_split='val_best_checkpoint')

    def forward_pass(self, batch):
        targets = batch[-1]  # the last entry of the batch tuple is always the targets
        gnn_out = self.model(*batch[0])  # foward the rest of the batch to the GNN
        predictions = self.critic(gnn_out)  # forward the GNN output to the critic
        return self.loss_func(predictions, targets), predictions, targets

    def process_batch(self, batch, optim):
        loss, predictions, targets = self.forward_pass(batch)
        if optim != None:  # run backpropagation if an optimizer is provided
            loss.backward()
            self.optim.step()
            self.after_optim_step()  # overwrite this function to do stuff before zeroing out grads
            self.optim.zero_grad()
            self.optim_steps += 1
        return loss, predictions.detach(), targets.detach()

    def initialize_optimizer(self, optim):
        self.optim = optim(self.critic.parameters(), **self.args.optimizer_critic_params)

    def save_model_state(self, epoch: int, checkpoint_name: str):
        torch.save({
            'epoch': epoch,
            'best_val_score': self.best_val_score,
            'optim_steps': self.optim_steps,
            'model_state_dict': self.model.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'scheduler_state_dict': None if self.lr_scheduler == None else self.lr_scheduler.state_dict()
        }, os.path.join(self.writer.log_dir, checkpoint_name))
