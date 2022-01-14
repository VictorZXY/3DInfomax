import copy

import ogb
import dgl
import torch
from dgl.dataloading import GraphDataLoader
from ogb.graphproppred import DglGraphPropPredDataset, Evaluator
from torch.nn import MSELoss, Sequential
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.pna import PNA
from models.base_layers import MLP


def _collate_fn(batch):
    # batch is a list of tuple (graph, label)
    graphs = [e[0] for e in batch]
    g = dgl.batch(graphs)
    labels = [e[1] for e in batch]
    labels = torch.stack(labels, 0)
    return g, labels


def pretrain_baseline(modelA, modelB, critic, num_epochs, dataset,
                      model_optimiser='Adam', model_optimiser_kwargs={},
                      critic_optimiser='Adam', critic_optimiser_kwargs={},
                      loss='MSELoss', device='cuda', tensorboard_writer=None):
    # load TensorBoard writer
    tensorboard_writer = tensorboard_writer or SummaryWriter()

    # load dataset
    dataset = DglGraphPropPredDataset(name=dataset)
    split_idx = dataset.get_idx_split()

    # DataLoader
    train_loader = GraphDataLoader(dataset[split_idx["train"]], batch_size=32,
                                   shuffle=True, collate_fn=_collate_fn)
    valid_loader = GraphDataLoader(dataset[split_idx["valid"]], batch_size=32,
                                   shuffle=False, collate_fn=_collate_fn)
    test_loader = GraphDataLoader(dataset[split_idx["test"]], batch_size=32,
                                  shuffle=False, collate_fn=_collate_fn)

    # load models onto device
    modelA = modelA.to(device)
    modelB = modelB.to(device)
    critic = critic.to(device)

    # load optimisers
    if model_optimiser == 'Adam':
        modelA_optim = Adam(modelA.parameters(), **model_optimiser_kwargs)
        modelB_optim = Adam(modelB.parameters(), **model_optimiser_kwargs)
    else:
        assert False
    if critic_optimiser == 'Adam':
        critic_optim = Adam(critic.parameters(), **critic_optimiser_kwargs)
    else:
        assert False

    # load loss function
    if loss == 'MSELoss':
        loss = MSELoss()
    else:
        assert False

    # pre-training
    train_loader_size = len(train_loader)
    for epoch in range(num_epochs):
        batch_i = 0
        for batch in tqdm(train_loader):
            graph = batch[0].to(device)
            graph.ndata['feat'] = graph.ndata['feat'].long()
            graph.edata['feat'] = graph.edata['feat'].long()
            graph_copy = copy.deepcopy(graph)

            modelA_out = modelA(graph)
            modelB_out = modelB(graph_copy)

            criticA_out = critic(modelA_out)
            criticB_out = critic(modelB_out)

            lossAB = loss(criticA_out, modelB_out)
            lossBA = loss(criticB_out, modelA_out)
            modelA_loss = lossAB - lossBA
            modelB_loss = lossBA - lossAB
            critic_loss = lossAB + lossBA

            tensorboard_writer.add_scalar('Pretrain_loss/GNN_A', modelA_loss,
                                          epoch * train_loader_size + batch_i)
            tensorboard_writer.add_scalar('Pretrain_loss/GNN_B', modelB_loss,
                                          epoch * train_loader_size + batch_i)
            tensorboard_writer.add_scalar('Pretrain_loss/Critic', critic_loss,
                                          epoch * train_loader_size + batch_i)
            batch_i += 1

            modelA_loss.backward(inputs=list(modelA.parameters()),
                                 retain_graph=True)
            modelA_optim.step()
            modelB_loss.backward(inputs=list(modelB.parameters()),
                                 retain_graph=True)
            modelB_optim.step()
            critic_loss.backward(inputs=list(critic.parameters()))
            critic_optim.step()

            modelA_optim.zero_grad()
            modelB_optim.zero_grad()
            critic_optim.zero_grad()

        print(
            f'Epoch {epoch + 1}: modelA loss: {modelA_loss:.4}, modelB loss: {modelB_loss:.4}, critic loss: {critic_loss:.4}')

    # save model
    tensorboard_writer.flush()
    if modelA_loss < modelB_loss:
        torch.save({
            'model_state_dict': modelA.state_dict(),
        }, 'trained_models/pretrained_baseline.pt')
    else:
        torch.save({
            'model_state_dict': modelB.state_dict(),
        }, 'trained_models/pretrained_baseline.pt')


def pretrain_baseline_2_critics(modelA, modelB, criticA, criticB, num_epochs,
                                dataset, model_optimiser='Adam',
                                model_optimiser_kwargs={},
                                critic_optimiser='Adam',
                                critic_optimiser_kwargs={},
                                loss='MSELoss', device='cuda',
                                tensorboard_writer=None):
    # load TensorBoard writer
    tensorboard_writer = tensorboard_writer or SummaryWriter()

    # load dataset
    dataset = DglGraphPropPredDataset(name=dataset)
    split_idx = dataset.get_idx_split()

    # DataLoader
    train_loader = GraphDataLoader(dataset[split_idx["train"]], batch_size=32,
                                   shuffle=True, collate_fn=_collate_fn)
    valid_loader = GraphDataLoader(dataset[split_idx["valid"]], batch_size=32,
                                   shuffle=False, collate_fn=_collate_fn)
    test_loader = GraphDataLoader(dataset[split_idx["test"]], batch_size=32,
                                  shuffle=False, collate_fn=_collate_fn)

    # load models onto device
    modelA = modelA.to(device)
    modelB = modelB.to(device)
    criticA = criticA.to(device)
    criticB = criticB.to(device)

    # load optimisers
    if model_optimiser == 'Adam':
        modelA_optim = Adam(modelA.parameters(), **model_optimiser_kwargs)
        modelB_optim = Adam(modelB.parameters(), **model_optimiser_kwargs)
    else:
        assert False
    if critic_optimiser == 'Adam':
        criticA_optim = Adam(criticA.parameters(), **critic_optimiser_kwargs)
        criticB_optim = Adam(criticB.parameters(), **critic_optimiser_kwargs)
    else:
        assert False

    # load loss function
    if loss == 'MSELoss':
        loss = MSELoss()
    else:
        assert False

    # pre-training
    train_loader_size = len(train_loader)
    for epoch in range(num_epochs):
        batch_i = 0
        for batch in tqdm(train_loader):
            graph = batch[0].to(device)
            graph.ndata['feat'] = graph.ndata['feat'].long()
            graph.edata['feat'] = graph.edata['feat'].long()
            graph_copy = copy.deepcopy(graph)

            modelA_out = modelA(graph)
            modelB_out = modelB(graph_copy)

            criticA_out = criticA(modelA_out)
            criticB_out = criticB(modelB_out)

            lossAB = loss(criticA_out, modelB_out)
            lossBA = loss(criticB_out, modelA_out)
            modelA_loss = lossAB - lossBA
            modelB_loss = lossBA - lossAB
            criticA_loss = lossAB
            criticB_loss = lossBA

            tensorboard_writer.add_scalar('Pretrain_loss/GNN_A', modelA_loss,
                                          epoch * train_loader_size + batch_i)
            tensorboard_writer.add_scalar('Pretrain_loss/GNN_B', modelB_loss,
                                          epoch * train_loader_size + batch_i)
            tensorboard_writer.add_scalar('Pretrain_loss/Critic_A', criticA_loss,
                                          epoch * train_loader_size + batch_i)
            tensorboard_writer.add_scalar('Pretrain_loss/Critic_B', criticB_loss,
                                          epoch * train_loader_size + batch_i)
            batch_i += 1

            modelA_loss.backward(inputs=list(modelA.parameters()),
                                 retain_graph=True)
            modelA_optim.step()
            modelB_loss.backward(inputs=list(modelB.parameters()),
                                 retain_graph=True)
            modelB_optim.step()
            criticA_loss.backward(inputs=list(criticA.parameters()))
            criticA_optim.step()
            criticB_loss.backward(inputs=list(criticB.parameters()))
            criticB_optim.step()

            modelA_optim.zero_grad()
            modelB_optim.zero_grad()
            criticA_optim.zero_grad()
            criticB_optim.zero_grad()

        print(
            f'Epoch {epoch + 1}: modelA loss: {modelA_loss:.4}, modelB loss: {modelB_loss:.4}, criticA loss: {criticA_loss:.4}, criticB loss: {criticB_loss:.4}')

    # save model
    tensorboard_writer.flush()
    if modelA_loss < modelB_loss:
        torch.save({
            'model_state_dict': modelA.state_dict(),
        }, 'trained_models/pretrained_baseline.pt')
    else:
        torch.save({
            'model_state_dict': modelB.state_dict(),
        }, 'trained_models/pretrained_baseline.pt')


def finetune_baseline_vs_control(pretrained_model, control_model, num_epochs,
                                 dataset, optimiser='Adam', optimiser_kwargs={},
                                 loss='MSELoss', device='cuda',
                                 tensorboard_writer=None):
    # load TensorBoard writer
    tensorboard_writer = tensorboard_writer or SummaryWriter()

    # load evaluator
    evaluator = Evaluator(name=dataset)

    # load dataset
    dataset = DglGraphPropPredDataset(name=dataset)
    split_idx = dataset.get_idx_split()

    # DataLoader
    train_loader = GraphDataLoader(dataset[split_idx["train"]], batch_size=32,
                                   shuffle=True, collate_fn=_collate_fn)
    valid_loader = GraphDataLoader(dataset[split_idx["valid"]], batch_size=32,
                                   shuffle=False, collate_fn=_collate_fn)
    test_loader = GraphDataLoader(dataset[split_idx["test"]], batch_size=32,
                                  shuffle=False, collate_fn=_collate_fn)

    # load models onto device
    pretrained_model = pretrained_model.to(device)
    control_model = control_model.to(device)

    # load optimisers
    if optimiser == 'Adam':
        pretrained_model_optim = Adam(pretrained_model.parameters(),
                                      **optimiser_kwargs)
        control_model_optim = Adam(control_model.parameters(),
                                   **optimiser_kwargs)
    else:
        assert False

    # load loss function
    if loss == 'MSELoss':
        loss = MSELoss()
    else:
        assert False

    # fine-tuning both the pre-trained model and the control model
    train_loader_size = len(train_loader)
    for epoch in range(num_epochs):
        batch_i = 0
        for batch in tqdm(train_loader):
            graph = batch[0].to(device)
            graph.ndata['feat'] = graph.ndata['feat'].long()
            graph.edata['feat'] = graph.edata['feat'].long()
            graph_copy = copy.deepcopy(graph)

            pretrained_model_out = pretrained_model(graph)
            control_model_out = control_model(graph_copy)

            label = batch[1].to(device)

            pretrained_model_loss = loss(pretrained_model_out, label)
            control_model_loss = loss(control_model_out, label)

            tensorboard_writer.add_scalar('Finetune_loss', {
                'Baseline': pretrained_model_loss,
                'Control': control_model_loss,
            }, epoch * train_loader_size + batch_i)

            pretrained_model_loss.backward()
            pretrained_model_optim.step()
            control_model_loss.backward()
            control_model_optim.step()

            pretrained_model_optim.zero_grad()
            control_model_optim.zero_grad()

            pretrained_model_result_dict = evaluator.eval(
                {'y_true': label, 'y_pred': pretrained_model_out})
            control_model_result_dict = evaluator.eval(
                {'y_true': label, 'y_pred': control_model_out})

        print(
            f'Epoch {epoch + 1}: pretrained model loss: {pretrained_model_loss:.4}, control model loss: {control_model_loss:.4}')


if __name__ == '__main__':
    modelA = PNA(hidden_dim=20,
                 target_dim=20,
                 aggregators=['sum'],
                 scalers=['identity'],
                 readout_aggregators=['sum'],
                 readout_batchnor=True,
                 readout_hidden_dim=None,
                 readout_layer=2,
                 residual=True,
                 pairwise_distances=False,
                 activation='relu')

    modelB = PNA(hidden_dim=20,
                 target_dim=20,
                 aggregators=['sum'],
                 scalers=['identity'],
                 readout_aggregators=['sum'],
                 readout_batchnor=True,
                 readout_hidden_dim=None,
                 readout_layer=2,
                 residual=True,
                 pairwise_distances=False,
                 activation='relu')

    critic = MLP(in_dim=20, out_dim=20, layers=1)

    criticA = MLP(in_dim=20, out_dim=20, layers=1)
    criticB = copy.deepcopy(critic)

    # modelA = PNA(hidden_dim=200,
    #              target_dim=256,
    #              aggregators=['mean', 'max', 'min', 'std'],
    #              scalers=['identity', 'amplification', 'attenuation'],
    #              readout_aggregators=['min', 'max', 'mean'],
    #              readout_batchnor=True,
    #              readout_hidden_dim=200,
    #              readout_layer=2,
    #              residual=True,
    #              pairwise_distances=False,
    #              activation='relu')
    #
    # modelB = PNA(hidden_dim=100,
    #              target_dim=256,
    #              aggregators=['mean', 'max', 'min', 'std'],
    #              scalers=['identity', 'amplification', 'attenuation'],
    #              readout_aggregators=['min', 'max', 'mean'],
    #              readout_batchnor=True,
    #              readout_hidden_dim=200,
    #              readout_layer=2,
    #              residual=True,
    #              pairwise_distances=False,
    #              activation='relu')
    #
    # critic = MLP(in_dim=256, out_dim=256, layers=1)

    tensorboard_writer = SummaryWriter()

    pretrain_baseline(modelA, modelB, critic, num_epochs=10,
                      model_optimiser_kwargs={'lr': 1e-5},
                      critic_optimiser_kwargs={'lr': 1e-5},
                      dataset='ogbg-molhiv',
                      tensorboard_writer=tensorboard_writer)

    # pretrain_baseline_2_critics(modelA, modelB, criticA, criticB, num_epochs=10,
    #                             model_optimiser_kwargs={'lr': 1e-5},
    #                             critic_optimiser_kwargs={'lr': 1e-5},
    #                             dataset='ogbg-molhiv',
    #                             tensorboard_writer=tensorboard_writer)

    pretrained_model = PNA(hidden_dim=20,
                           target_dim=20,
                           aggregators=['sum'],
                           scalers=['identity'],
                           readout_aggregators=['sum'],
                           readout_batchnor=True,
                           readout_hidden_dim=None,
                           readout_layer=2,
                           residual=True,
                           pairwise_distances=False,
                           activation='relu')
    checkpoint = torch.load('trained_models/pretrained_baseline.pt')
    pretrained_model.load_state_dict(checkpoint['model_state_dict'])
    pretrained_model = Sequential(
        pretrained_model,
        MLP(in_dim=20, out_dim=1, layers=1)
    )

    control_model = PNA(hidden_dim=20,
                        target_dim=20,
                        aggregators=['sum'],
                        scalers=['identity'],
                        readout_aggregators=['sum'],
                        readout_batchnor=True,
                        readout_hidden_dim=None,
                        readout_layer=2,
                        residual=True,
                        pairwise_distances=False,
                        activation='relu')
    control_model = Sequential(
        control_model,
        MLP(in_dim=20, out_dim=1, layers=1)
    )

    finetune_baseline_vs_control(pretrained_model, control_model, num_epochs=10,
                                 optimiser_kwargs={'lr': 1e-5},
                                 dataset='ogbg-molesol',
                                 tensorboard_writer=tensorboard_writer)
