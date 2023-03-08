import copy
import os
import shutil
from typing import Dict, Callable

import torch
from torch.utils.data import DataLoader

from commons.utils import tensorboard_singular_value_plot
from trainer.trainer import Trainer


class CLASSTrainer(Trainer):
    def __init__(self, model, model2, critic, critic2, args, metrics: Dict[str, Callable], main_metric: str,
                 device: torch.device, tensorboard_functions: Dict[str, Callable],
                 optim=None, main_metric_goal: str = 'min', loss_func=torch.nn.MSELoss,
                 scheduler_step_per_batch: bool = True, **kwargs):
        # move to device before loading optim params in super class
        # no need to move model because it will be moved in super class call
        self.model2 = model2.to(device)
        self.critic = critic.to(device)
        self.critic2 = critic2.to(device)
        super(CLASSTrainer, self).__init__(model, args, metrics, main_metric, device, tensorboard_functions,
                                           optim, main_metric_goal, loss_func, scheduler_step_per_batch)

        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location=self.device)
            self.model2.load_state_dict(checkpoint['model2_state_dict'])

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
        graph = tuple(batch)[0]
        graph_copy = copy.deepcopy(graph)
        modelA_out = self.model(graph)  # forward the rest of the batch to the model
        modelB_out = self.model2(graph_copy)  # forward the rest of the batch to the model
        criticA_out = self.critic(modelA_out)
        criticB_out = self.critic2(modelB_out)

        return self.loss_func(modelA_out, modelB_out, criticA_out, criticB_out), modelA_out, modelB_out

    def process_batch(self, batch, optim):
        loss, predictions, targets = self.forward_pass(batch)
        modelA_loss, modelB_loss, criticA_loss, criticB_loss, loss_components = loss

        if optim != None:  # run backpropagation if an optimizer is provided
            if self.args.iterations_per_model == 0:
                modelA_loss.backward(inputs=list(self.model.parameters()), retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                self.optim.step()
                modelB_loss.backward(inputs=list(self.model2.parameters()), retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.model2.parameters(), max_norm=1)
                self.optim2.step()
                criticA_loss.backward(inputs=list(self.critic.parameters()))
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1)
                self.optim_critic.step()
                criticB_loss.backward(inputs=list(self.critic2.parameters()))
                torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1)
                self.optim_critic2.step()

                self.optim.zero_grad()
                self.optim2.zero_grad()
                self.optim_critic.zero_grad()
                self.optim_critic2.zero_grad()
                self.optim_steps += 1
            else:
                if (self.optim_steps // self.args.iterations_per_model) % 2 == 0:
                    modelA_loss.backward(inputs=list(self.model.parameters()), retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                    self.optim.step()
                    self.optim.zero_grad()
                else:
                    modelB_loss.backward(inputs=list(self.model2.parameters()), retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(self.model2.parameters(), max_norm=1)
                    self.optim2.step()
                    self.optim2.zero_grad()

                criticA_loss.backward(inputs=list(self.critic.parameters()))
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1)
                self.optim_critic.step()
                criticB_loss.backward(inputs=list(self.critic2.parameters()))
                torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1)
                self.optim_critic2.step()

                self.optim_critic.zero_grad()
                self.optim_critic2.zero_grad()
                self.optim_steps += 1

        return modelA_loss, (predictions.detach()), (targets.detach())

    def run_per_epoch_evaluations(self, data_loader):
        pass
        # for loader, loader_name in data_loader:
        #     print(f'computing PCA explained variance of the {loader_name} loader outputs')
        #
        #     predictions = []
        #     targets = []
        #     for batch in loader:
        #         batch = [element.to(self.device) for element in batch]
        #         _, modelA_out, modelB_out = self.process_batch(batch, optim=None)
        #         predictions.append(modelA_out)
        #         targets.append(modelB_out)
        #     predictions = torch.cat(predictions, dim=0)
        #     targets = torch.cat(targets, dim=0)
        #
        #     for X, data_split in [(predictions, f'{loader_name}_pred'), (targets, f'{loader_name}_targets')]:
        #         tensorboard_singular_value_plot(predictions=X, targets=None, writer=self.writer, step=self.optim_steps,
        #                                         data_split=data_split)
        #
        #     print(f'finish computing PCA explained variance of the {loader_name} loader outputs')

    def initialize_optimizer(self, optim):
        self.optim = optim(self.model.parameters(), **self.args.optimizer_params)
        self.optim2 = optim(self.model2.parameters(), **self.args.optimizer2_params)
        self.optim_critic = optim(self.critic.parameters(), **self.args.optimizer_critic_params)
        self.optim_critic2 = optim(self.critic2.parameters(), **self.args.optimizer_critic2_params)

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


class CLASSHybridBarlowTwinsTrainer(Trainer):
    def __init__(self, model, model2, args, metrics: Dict[str, Callable], main_metric: str,
                 device: torch.device, tensorboard_functions: Dict[str, Callable],
                 optim=None, main_metric_goal: str = 'min', loss_func=torch.nn.MSELoss,
                 scheduler_step_per_batch: bool = True, **kwargs):
        # move to device before loading optim params in super class
        # no need to move model because it will be moved in super class call
        self.model2 = model2.to(device)
        super(CLASSHybridBarlowTwinsTrainer, self).__init__(model, args, metrics, main_metric, device,
                                                            tensorboard_functions, optim, main_metric_goal, loss_func,
                                                            scheduler_step_per_batch)

        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location=self.device)
            self.model2.load_state_dict(checkpoint['model2_state_dict'])

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
        graph = tuple(batch)[0]
        graph_copy = copy.deepcopy(graph)
        modelA_out = self.model(graph)  # forward the rest of the batch to the model
        modelB_out = self.model2(graph_copy)  # forward the rest of the batch to the model

        return self.loss_func(modelA_out, modelB_out), modelA_out, modelB_out

    def process_batch(self, batch, optim):
        loss, predictions, targets = self.forward_pass(batch)
        modelA_loss, modelB_loss, loss_components = loss

        if optim != None:  # run backpropagation if an optimizer is provided
            modelA_loss.backward(inputs=list(self.model.parameters()), retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            self.optim.step()
            modelB_loss.backward(inputs=list(self.model2.parameters()), retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.model2.parameters(), max_norm=1)
            self.optim2.step()

            self.optim.zero_grad()
            self.optim2.zero_grad()
            self.optim_steps += 1

        return modelA_loss - modelB_loss, (predictions.detach()), (targets.detach())

    def run_per_epoch_evaluations(self, data_loader):
        pass
        # for loader, loader_name in data_loader:
        #     print(f'computing PCA explained variance of the {loader_name} loader outputs')
        #
        #     predictions = []
        #     targets = []
        #     for batch in loader:
        #         batch = [element.to(self.device) for element in batch]
        #         _, modelA_out, modelB_out = self.process_batch(batch, optim=None)
        #         predictions.append(modelA_out)
        #         targets.append(modelB_out)
        #     predictions = torch.cat(predictions, dim=0)
        #     targets = torch.cat(targets, dim=0)
        #
        #     for X, data_split in [(predictions, f'{loader_name}_pred'), (targets, f'{loader_name}_targets')]:
        #         tensorboard_singular_value_plot(predictions=X, targets=None, writer=self.writer, step=self.optim_steps,
        #                                         data_split=data_split)
        #
        #     print(f'finish computing PCA explained variance of the {loader_name} loader outputs')

    def initialize_optimizer(self, optim):
        transferred_keys = [k for k in self.model.state_dict().keys() if
                            any(transfer_layer in k for transfer_layer in self.args.transfer_layers) and
                            not any(to_exclude in k for to_exclude in self.args.exclude_from_transfer)]
        frozen_keys = [k for k in self.model.state_dict().keys() if
                       any(to_freeze in k for to_freeze in self.args.frozen_layers)]
        frozen_params = [v for k, v in self.model.named_parameters() if k in frozen_keys]
        transferred_params = [v for k, v in self.model.named_parameters() if k in transferred_keys]
        new_params = [v for k, v in self.model.named_parameters() if
                      k not in transferred_keys and 'batch_norm' not in k and k not in frozen_keys]
        batch_norm_params = [v for k, v in self.model.named_parameters() if
                             'batch_norm' in k and k not in transferred_keys and k not in frozen_keys]

        transfer_lr = self.args.optimizer_params['lr'] if self.args.transferred_lr == None else self.args.transferred_lr
        # the order of the params here determines in which order they will start being updated during warmup when using ordered warmup in the warmupwrapper
        param_groups = []
        if batch_norm_params != []:
            param_groups.append({'params': batch_norm_params, 'weight_decay': 0})
        param_groups.append({'params': new_params})
        if transferred_params != []:
            param_groups.append({'params': transferred_params, 'lr': transfer_lr})
        if frozen_params != []:
            param_groups.append({'params': frozen_params, 'lr': 0})
        self.optim = optim(param_groups, **self.args.optimizer_params)

        transferred_keys2 = [k for k in self.model2.state_dict().keys() if
                             any(transfer_layer in k for transfer_layer in self.args.transfer_layers2) and
                             not any(to_exclude in k for to_exclude in self.args.exclude_from_transfer2)]
        frozen_keys2 = [k for k in self.model2.state_dict().keys() if
                        any(to_freeze in k for to_freeze in self.args.frozen_layers2)]
        frozen_params2 = [v for k, v in self.model2.named_parameters() if k in frozen_keys2]
        transferred_params2 = [v for k, v in self.model2.named_parameters() if k in transferred_keys2]
        new_params2 = [v for k, v in self.model2.named_parameters() if
                       k not in transferred_keys2 and 'batch_norm' not in k and k not in frozen_keys2]
        batch_norm_params2 = [v for k, v in self.model2.named_parameters() if
                              'batch_norm' in k and k not in transferred_keys2 and k not in frozen_keys2]

        transfer_lr2 = self.args.optimizer2_params['lr'] if self.args.transferred_lr2 == None else self.args.transferred_lr2
        # the order of the params here determines in which order they will start being updated during warmup when using ordered warmup in the warmupwrapper
        param_groups2 = []
        if batch_norm_params2 != []:
            param_groups2.append({'params': batch_norm_params2, 'weight_decay': 0})
        param_groups2.append({'params': new_params2})
        if transferred_params2 != []:
            param_groups2.append({'params': transferred_params2, 'lr': transfer_lr2})
        if frozen_params2 != []:
            param_groups2.append({'params': frozen_params2, 'lr': 0})
        self.optim2 = optim(param_groups2, **self.args.optimizer2_params)

    def save_model_state(self, epoch: int, checkpoint_name: str):
        torch.save({
            'epoch': epoch,
            'best_val_score': self.best_val_score,
            'optim_steps': self.optim_steps,
            'model_state_dict': self.model.state_dict(),
            'model2_state_dict': self.model2.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'optimizer2_state_dict': self.optim2.state_dict(),
            'scheduler_state_dict': None if self.lr_scheduler == None else self.lr_scheduler.state_dict()
        }, os.path.join(self.writer.log_dir, checkpoint_name))
