import csv
import os
import pickle
import time

import dgl
from torch.utils.data import Dataset


class ZINCMoleculeDGL(Dataset):
    def __init__(self, data_dir, split, num_graphs=None):
        self.data_dir = data_dir
        self.split = split
        self.num_graphs = num_graphs

        with open(data_dir + "/%s.pickle" % self.split, "rb") as f:
            self.data = pickle.load(f)

        if self.num_graphs in [10000, 1000]:
            # loading the sampled indices from file ./molecules/<split>.index
            with open(data_dir + "/%s.index" % self.split, "r") as f:
                data_idx = [list(map(int, idx)) for idx in csv.reader(f)]
                self.data = [self.data[i] for i in data_idx[0]]

            assert len(self.data) == num_graphs, "Sample num_graphs again; available idx: train/val/test => 10k/1k/1k"

        """
        data is a list of Molecule dict objects with following attributes

          molecule = data[idx]
        ; molecule['num_atom'] : nb of atoms, an integer (N)
        ; molecule['atom_type'] : tensor of size N, each element is an atom type, an integer between 0 and num_atom_type
        ; molecule['bond_type'] : tensor of size N x N, each element is a bond type, an integer between 0 and num_bond_type
        ; molecule['logP_SA_cycle_normalized'] : the chemical property to regress, a float variable
        """

        self.graph_lists = []
        self.graph_labels = []
        self.n_samples = len(self.data)
        self._prepare()

    def _prepare(self):
        print("preparing %d graphs for the %s set..." % (self.num_graphs, self.split.upper()))

        for molecule in self.data:
            node_features = molecule['atom_type'].unsqueeze(-1).long()

            adj = molecule['bond_type']
            edge_list = (adj != 0).nonzero()  # converting adj matrix to edge_list

            edge_idxs_in_adj = edge_list.split(1, dim=1)
            edge_features = adj[edge_idxs_in_adj].long()

            # Create the DGL Graph
            g = dgl.DGLGraph()
            g.add_nodes(molecule['num_atom'])
            g.ndata['feat'] = node_features

            for src, dst in edge_list:
                g.add_edges(src.item(), dst.item())
            g.edata['feat'] = edge_features

            self.graph_lists.append(g)
            self.graph_labels.append(molecule['logP_SA_cycle_normalized'])

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx]


class ZINCDataset(Dataset):
    def __init__(self, data_dir='dataset', name='ZINC-full'):
        t0 = time.time()
        self.data_dir = os.path.join(data_dir, 'ZINC/molecules')
        self.name = name

        self.num_atom_type = 28  # known meta-info about the zinc dataset; can be calculated as well
        self.num_bond_type = 4  # known meta-info about the zinc dataset; can be calculated as well

        if self.name == 'ZINC-full':
            self.data_dir = os.path.join(self.data_dir, 'zinc_full')
            self.train = ZINCMoleculeDGL(self.data_dir, 'train', num_graphs=220011)
            self.val = ZINCMoleculeDGL(self.data_dir, 'val', num_graphs=24445)
            self.test = ZINCMoleculeDGL(self.data_dir, 'test', num_graphs=5000)
        else:
            self.train = ZINCMoleculeDGL(self.data_dir, 'train', num_graphs=10000)
            self.val = ZINCMoleculeDGL(self.data_dir, 'val', num_graphs=1000)
            self.test = ZINCMoleculeDGL(self.data_dir, 'test', num_graphs=1000)
        print("Time taken: {:.4f}s".format(time.time() - t0))
