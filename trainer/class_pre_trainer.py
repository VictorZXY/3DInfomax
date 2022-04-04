import copy
import os
import shutil
from itertools import chain
from typing import Union, Tuple, Dict, Callable

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from commons.utils import move_to_device, log
from trainer.self_supervised_trainer import SelfSupervisedTrainer
from trainer.trainer import Trainer


class CLASSTrainer(Trainer):
    def __init__(self, model, model2, critic, critic2, args, metrics: Dict[str, Callable], main_metric: str,
                 device: torch.device, tensorboard_functions: Dict[str, Callable], decoder=None, decoder2=None,
                 optim=None, main_metric_goal: str = 'min', loss_func=torch.nn.MSELoss,
                 scheduler_step_per_batch: bool = True, **kwargs):
        # move to device before loading optim params in super class
        # no need to move model because it will be moved in super class call
        self.model2 = model2.to(device)
        self.critic = critic.to(device)
        self.critic2 = critic2.to(device)
        self.decoder = decoder.to(device) if decoder else None
        self.decoder2 = decoder2.to(device) if decoder2 else None
        super(CLASSTrainer, self).__init__(model, args, metrics, main_metric, device, tensorboard_functions,
                                           optim, main_metric_goal, loss_func, scheduler_step_per_batch)

        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location=self.device)
            self.model2.load_state_dict(checkpoint['model2_state_dict'])

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        epochs_no_improve = 0  # counts every epoch that the validation accuracy did not improve for early stopping
        for epoch in range(self.start_epoch, self.args.num_epochs + 1):  # loop over the dataset multiple times
            self.epoch = epoch
            self.model.train()
            self.predict(train_loader, optim=self.optim)

            self.model.eval()
            with torch.no_grad():
                metrics, _, _ = self.predict(val_loader)
                val_score = metrics[self.main_metric]

                if self.lr_scheduler != None and not self.scheduler_step_per_batch:
                    self.step_schedulers(metrics=val_score)

                if self.args.eval_per_epochs > 0 and epoch % self.args.eval_per_epochs == 0:
                    self.run_per_epoch_evaluations([(train_loader, 'train'), (val_loader, 'val')])

                self.tensorboard_log(metrics, data_split='val', log_hparam=True, step=self.optim_steps)
                val_loss = metrics[type(self.loss_func).__name__]
                log(f'[Epoch {epoch}] {self.main_metric}: {val_score} val loss: {val_loss}')
                # save the model with the best main_metric depending on wether we want to maximize or minimize the main metric
                if val_score >= self.best_val_score and self.main_metric_goal == 'max' or val_score <= self.best_val_score and self.main_metric_goal == 'min':
                    epochs_no_improve = 0
                    self.best_val_score = val_score
                    self.save_checkpoint(epoch, checkpoint_name='best_checkpoint.pt')
                else:
                    epochs_no_improve += 1
                self.save_checkpoint(epoch, checkpoint_name='last_checkpoint.pt')
                log('Epochs with no improvement: [', epochs_no_improve, '] and the best  ', self.main_metric,
                    ' was in ', epoch - epochs_no_improve)
                if epochs_no_improve >= self.args.patience and epoch >= self.args.minimum_epochs:  # stopping criterion
                    log(f'Early stopping criterion based on -{self.main_metric}- that should be {self.main_metric_goal}-imized reached after {epoch} epochs. Best model checkpoint was in epoch {epoch - epochs_no_improve}.')
                    break
                if epoch in self.args.models_to_save:
                    shutil.copyfile(os.path.join(self.writer.log_dir, 'best_checkpoint.pt'),
                                    os.path.join(self.writer.log_dir, f'best_checkpoint_{epoch}epochs.pt'))
                self.after_epoch()
                #if val_loss > 10000:
                #    raise Exception

        # evaluate on best checkpoint
        checkpoint = torch.load(os.path.join(self.writer.log_dir, 'best_checkpoint.pt'), map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return self.evaluation(val_loader, data_split='val_best_checkpoint')

    def forward_pass(self, batch):
        graph = tuple(batch)[0]
        graph_copy = copy.deepcopy(graph)
        modelA_node_features, modelA_out = self.model(graph)  # foward the rest of the batch to the model
        modelB_node_features, modelB_out = self.model2(graph_copy)  # foward the rest of the batch to the model
        criticA_out = self.critic(modelA_out)
        criticB_out = self.critic2(modelB_out)
        decoderA_out = self.decoder(modelA_node_features) if self.decoder else None
        decoderB_out = self.decoder2(modelB_node_features) if self.decoder2 else None
        modelA_loss, modelB_loss, criticA_loss, criticB_loss, decoderA_loss, decoderB_loss, loss_components = self.loss_func(
            modelA_out, modelB_out, criticA_out, criticB_out, decoderA_out, decoderB_out, graph, graph_copy,
            output_regularisation=self.args.output_regularisation, coop_loss_coeff=self.args.coop_loss_coeff,
            adv_loss_coeff=self.args.adv_loss_coeff, device=self.device)

        return modelA_loss, modelB_loss, criticA_loss, criticB_loss, decoderA_loss, decoderB_loss, \
               (loss_components if loss_components != [] else None), modelA_out, modelB_out

    def process_batch(self, batch, optim):
        modelA_loss, modelB_loss, criticA_loss, criticB_loss, decoderA_loss, decoderB_loss, loss_components, predictions, targets = self.forward_pass(batch)

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

        return modelA_loss, loss_components, (predictions.detach()), (targets.detach())

    def run_per_epoch_evaluations(self, data_loader):
        for loader, loader_name in data_loader:
            print(f'computing PCA explained variance of the {loader_name} loader outputs')
            representations = []
            targets = []
            for batch in loader:
                batch = [element.to(self.device) for element in batch]
                _, _, modelA_out, modelB_out = self.process_batch(batch, optim=None)
                representations.append(modelA_out)
                targets.append(modelB_out)
            representations = torch.cat(representations, dim=0)
            targets = torch.cat(targets, dim=0)
            for n_components in [2, 4, 8]:
                for output_name, X in [('pred', representations), ('targets', targets)]:
                    pca = PCA(n_components=n_components)
                    pca.fit_transform(X.cpu())
                    total_explained_var_ratio = np.sum(pca.explained_variance_ratio_)
                    self.writer.add_scalar(f'PCA{n_components}_explained_variance_{loader_name}_{output_name}', total_explained_var_ratio, self.optim_steps)
            print(f'finish computing PCA explained variance of the {loader_name} loader outputs')

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
