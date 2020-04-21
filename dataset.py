import os
import pathlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, BatchSampler
from torch.utils.data import WeightedRandomSampler
import scipy.sparse as ssp
import sklearn.preprocessing as prep
import sklearn.pipeline as ppln
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

from utils import to_sparse_tensor, bin_data, normalize, split_bucketed_data, \
    summarize_feature_matrix


def get_composite_transformer(n_quantiles):
    transformer = ppln.Pipeline([
        ("quantile", prep.QuantileTransformer(output_distribution="normal",
                                              n_quantiles=n_quantiles)),
        ("normalize", prep.StandardScaler())
    ])
    return transformer


class BinnedTransformer:

    def __init__(self, num_bins, create_transformer_f):
        self.num_bins = num_bins

        self.transformers = [create_transformer_f() for _ in range(num_bins)]

    def fit_transform(self, x_reg, x_class):
        transformed_x_reg = np.copy(x_reg)
        for bin_idx in range(self.num_bins):
            sample_idcs = x_class == bin_idx
            transformer = self.transformers[bin_idx]
            transformed_x_reg[sample_idcs] = transformer.fit_transform(
                transformed_x_reg[sample_idcs])
        return transformed_x_reg

    def inverse_transform(self, x_reg, x_class):
        x_reg = x_reg.reshape(-1)
        transformed_x_reg = np.copy(x_reg)
        for bin_idx in range(self.num_bins):
            sample_idcs = x_class == bin_idx
            if np.sum(sample_idcs) == 0: continue   # no sample of that class
            transformer = self.transformers[bin_idx]
            transformed_x_reg[sample_idcs] = transformer.inverse_transform(
                x_reg[sample_idcs].reshape(-1, 1)).reshape(-1)
        return transformed_x_reg


class GraphTopologicalData:

    def __init__(self, adj_matrix=None, unweighted_adj_matrix=None,
                 inc_matrix=None, inc_matrix_dense=None, edge_indices=None,
                 edge_weights=None):
        self.adj_matrix = adj_matrix                        # NxN sparse matrix
        self.unweighted_adj_matrix = unweighted_adj_matrix  # NxN sparse matrix
        self.inc_matrix = inc_matrix                        # NxE sparse matrix
        self.inc_matrix_dense = inc_matrix_dense            # NxE dense matrix
        self.edge_indices = edge_indices                    # Ex2 dense matrix
        self.edge_weights = edge_weights                    # E dense vector


class UrbanPlanningDataset:

    def __init__(self, data_base_path="Data/", num_bins=4, batch_size=32,
                 n_quantiles=1000, resample=False,
                 excluded_node_feature_columns=tuple(),
                 excluded_edge_feature_columns=tuple(),
                 use_binned_transformer=False, include_approx_flows=False,
                 flow_adj_threshold=0, seed=7):
        """
        Loads city data set.
        :param data_base_path: Location at which to find the node features,
        edge features, and the adjacency matrix.
        :param num_bins: Number of bins for dividing the data set labels. The
        bin index may be a classification target or for computing MAEs for each
        bin separately.
        :param batch_size:
        :param n_quantiles: Number of quantiles to use for the quantile
        transformer that preprocesses features and labels.
        :param excluded_node_feature_columns: Tuple of names of the columns
        to remove from the node feature data set.
        :param excluded_edge_feature_columns: Tuple of names of the columns to
        remove from the edge feature data set.
        :param resample: If True, we use a weighted random sampler to ensure
        that each epoch contains an equal number of samples from each bin.
        :param use_binned_transformer: If True, the edge labels are rescaled
        using an individual transformer for each bin. Inverting the
        transformation then requires both a regression and classification
        prediction.
        :param include_approx_flows: If True, the edge features include the
        approximate flows (normally used just for flow adjacency matrix).
        :param flow_adj_threshold: When constructing the unweighted flow
        adjacency matrix, only include edges with a flow greater or equal that
        threshold.
        :param seed: Random seed to always obtain the same split into training,
        validation, and test set.
        :return: Tuple consisting of
            - Node features of shape [N, K]
            - Sparse adjacency matrix of shape [N, N]
            - Loader for the training set of edges
            - Loader for the validation set of edges
            - Loader for the test set of edges
            - Number of node features
            - Number of edge features
            - Scaler used for edge labels
        """
        print("Loading data")

        self.num_bins = num_bins
        self.batch_size = batch_size
        self.n_quantiles = n_quantiles
        self.use_binned_transformer = use_binned_transformer

        get_composite_transformer_f = lambda: get_composite_transformer(
            n_quantiles=n_quantiles)

        # Load node data
        (self.node_feats, self.num_nodes, self.num_node_feats,
         self.node_scaler) = self._load_node_data(data_base_path,
                                                  get_composite_transformer_f,
                                                  excluded_node_feature_columns)

        # Load edge data
        (flow_edge_indices, self.edge_feats, self.edge_labels,
         self.edge_labels_unscaled, self.label_scaler, self.edge_scaler,
         self.num_edges, self.num_edge_feats) = self._load_edge_data(
            data_base_path,
            get_composite_transformer_f,
            include_approx_flows,
            excluded_edge_feature_columns)
        self.max_label = np.max(self.edge_labels_unscaled)
        print(f"\tMax label {self.max_label}")

        (train_idcs, val_idcs, test_idcs) = self._load_dataset_split(
            data_base_path)

        # Load flow graph data
        (flow_adj_matrix, flow_inc_matrix, flow_adj_indices,
         unweighted_flow_adj_matrix,
         flow_adj_values) = self._load_flow_graph_data(
            data_base_path, self.num_nodes, self.num_edges, flow_adj_threshold)
        self.flow_topology = GraphTopologicalData(
            adj_matrix=flow_adj_matrix,
            edge_indices=flow_adj_indices,
            unweighted_adj_matrix=unweighted_flow_adj_matrix,
            inc_matrix=flow_inc_matrix,
            edge_weights=flow_adj_values
        )

        # Load geographical graph data
        (geo_adj_matrix, geo_inc_matrix,
         geo_edge_indices, geo_adj_values) = self._load_geo_graph_data(
            data_base_path, self.num_nodes, self.num_edges, self.flow_topology)
        self.geo_topology = GraphTopologicalData(
            adj_matrix=geo_adj_matrix,
            inc_matrix=geo_inc_matrix,
            edge_indices=geo_edge_indices,
            edge_weights=geo_adj_values)

        # Load bin data
        self.bin_bounds = [10.0, 100.0, 1000.0, 10000.0]
        (self.edge_buckets, self.train_bin_weights, self.val_bin_weights,
         self.test_bin_weights) = self._load_bin_data(self.bin_bounds,
                                                      self.edge_labels_unscaled,
                                                      num_bins, train_idcs,
                                                      val_idcs, test_idcs)
        print(f"\tBin counts: {np.array([np.sum(self.edge_buckets == i) for i in range(num_bins)])}")
        print(f"\tTraining bin weights: {self.train_bin_weights}")
        print(f"\tValidation bin weights: {self.val_bin_weights}")
        print(f"\tTest bin weights: {self.test_bin_weights}")

        # If specified, use the binned transformer to transform labels
        if use_binned_transformer:
            self.label_scaler = BinnedTransformer(self.num_bins,
                                                  get_composite_transformer_f)
            self.edge_labels = self.label_scaler.fit_transform(
                self.edge_labels_unscaled.reshape(-1, 1), self.edge_buckets).reshape(-1)
            # plt.hist(self.edge_labels, bins=100)
            # plt.show()

        # Create edge feature matrix
        indices = flow_edge_indices.transpose(1, 0)
        values = self.edge_feats
        edge_feat_matrix = torch.sparse.FloatTensor(torch.from_numpy(indices), torch.from_numpy(values))
        self.edge_feat_matrix = edge_feat_matrix.to_dense()

        # Convert numpy arrays to tensors
        self.node_feats = torch.from_numpy(self.node_feats)
        self.edge_feats = torch.from_numpy(self.edge_feats)
        flow_edge_indices = torch.from_numpy(flow_edge_indices)
        self.flow_topology.edge_indices = torch.from_numpy(self.flow_topology.edge_indices)
        self.flow_topology.edge_weights = torch.from_numpy(self.flow_topology.edge_weights)
        self.geo_topology.edge_indices = torch.from_numpy(self.geo_topology.edge_indices)
        self.geo_topology.edge_weights = torch.from_numpy(self.geo_topology.edge_weights)
        self.edge_labels = torch.from_numpy(self.edge_labels)
        self.edge_labels_unscaled = torch.from_numpy(self.edge_labels_unscaled)
        self.edge_buckets = torch.from_numpy(self.edge_buckets)
        self.train_bin_weights = torch.from_numpy(self.train_bin_weights)
        self.val_bin_weights = torch.from_numpy(self.val_bin_weights)
        self.test_bin_weights = torch.from_numpy(self.test_bin_weights)
        # Matrices
        self.geo_topology.adj_matrix = to_sparse_tensor(normalize(self.geo_topology.adj_matrix))
        self.geo_topology.inc_matrix = to_sparse_tensor(self.geo_topology.inc_matrix)
        self.flow_topology.adj_matrix = to_sparse_tensor(self.flow_topology.adj_matrix)   # Sparse tensor of shape [N, N] containing the flow values between nodes.
        self.flow_topology.unweighted_adj_matrix = to_sparse_tensor(self.flow_topology.unweighted_adj_matrix)
        self.flow_topology.inc_matrix = to_sparse_tensor(self.flow_topology.inc_matrix)
        self._check_data_consistency()

        # Create data loaders
        (self.train_loader, self.val_loader,
         self.test_loader) = self._create_data_loaders(train_idcs, val_idcs,
                                                       test_idcs,
                                                       self.train_bin_weights,
                                                       flow_edge_indices,       # different from flow_graph_topology.edge_indices because of additional 0-flows
                                                       self.edge_feats,
                                                       self.edge_labels,
                                                       self.edge_buckets,
                                                       batch_size, resample,
                                                       seed)

        print("Finished loading data")

    def _check_data_consistency(self):
        tensors = [self.node_feats, self.edge_feats,
                   self.flow_topology.edge_indices,
                   self.geo_topology.edge_indices, self.edge_labels,
                   self.edge_labels_unscaled, self.edge_buckets,
                   self.train_bin_weights, self.val_bin_weights,
                   self.test_bin_weights, self.geo_topology.adj_matrix,
                   self.geo_topology.inc_matrix, self.flow_topology.adj_matrix,
                   self.flow_topology.unweighted_adj_matrix,
                   self.flow_topology.inc_matrix, self.edge_feat_matrix]
        print("Checking ", end="")
        for idx, tensor in enumerate(tensors):
            print(f"{idx}, ", end="")
            if (isinstance(tensor, torch.sparse.FloatTensor) or
                    isinstance(tensor, torch.sparse.LongTensor)):
                assert not torch.isnan(tensor.coalesce().indices()).any()
                assert not torch.isnan(tensor.coalesce().values()).any()
            else:
                assert not torch.isnan(tensor).any()
        print("done")

    def to(self, device):
        """
        Moves all tensors of the dataset that will not be iterated over in
        minibatch to the specified device.
        :param device: Device specifier.
        """
        self.node_feats = self.node_feats.to(device=device)
        self.edge_feats = self.edge_feats.to(device=device)
        self.flow_topology.edge_indices = self.flow_topology.edge_indices.to(device=device)
        self.geo_topology.edge_indices = self.geo_topology.edge_indices.to(device=device)
        self.train_bin_weights = self.train_bin_weights.to(device=device)
        self.geo_topology.adj_matrix = self.geo_topology.adj_matrix.to(device=device)
        self.geo_topology.inc_matrix = self.geo_topology.inc_matrix.to(device=device)
        self.geo_topology.edge_weights = self.geo_topology.edge_weights.to(device=device)
        self.flow_topology.adj_matrix = self.flow_topology.adj_matrix.to(device=device)
        self.flow_topology.unweighted_adj_matrix = self.flow_topology.unweighted_adj_matrix.to(
            device=device)
        self.flow_topology.inc_matrix = self.flow_topology.inc_matrix.to(device=device)
        self.flow_topology.edge_weights = self.flow_topology.edge_weights.to(device=device)
        self.edge_feat_matrix = self.edge_feat_matrix.to(device=device)

    @staticmethod
    def _load_node_data(data_base_path, get_composite_transformer_f,
                        excluded_columns):
        # Node features
        node_data = pd.read_pickle(os.path.join(data_base_path, "node_data.pk"))
        if len(excluded_columns) > 0:
            node_data.drop(list(excluded_columns), axis=1, inplace=True)
        node_feats = node_data.values
        # Rescale continuous features
        node_scaler = get_composite_transformer_f()
        cont_feature_idcs = UrbanPlanningDataset._get_continuous_feature_idcs(node_data)
        node_feats[:, cont_feature_idcs] = node_scaler.fit_transform(node_feats[:, cont_feature_idcs])
        node_feats = node_feats.astype(np.float32)
        num_nodes = node_feats.shape[0]
        num_node_feats = node_feats.shape[1]
        return node_feats, num_nodes, num_node_feats, node_scaler

    @staticmethod
    def _load_edge_data(data_base_path, get_composite_transformer_f,
                        include_approx_flows, excluded_columns):
        # Edge data
        edge_data = pd.read_pickle(os.path.join(data_base_path, "edge_data.pk"))
        if len(excluded_columns) > 0:
            edge_data.drop(list(excluded_columns), axis=1, inplace=True)
        edge_feats = edge_data.values
        edge_indices = edge_feats[:, :2].astype(np.int)
        edge_feats = edge_feats[:, 2:]
        # Load approximate flows and potentially concatenate to edge features
        # approx_flows = np.load(os.path.join(data_base_path,
        #                                     "approx_flows.npy"))
        if include_approx_flows:
            raise NotImplementedError
            # edge_feats = np.concatenate((edge_feats, approx_flows.reshape(-1, 1)),
            #                             axis=-1)
        num_edges = edge_feats.shape[0]
        edge_labels = np.load(os.path.join(data_base_path, "flows.npy"))
        edge_labels_unscaled = np.copy(edge_labels).astype(np.float32)
        # Transform edge features
        edge_scaler = get_composite_transformer_f()
        cont_feature_idcs = UrbanPlanningDataset._get_continuous_feature_idcs(edge_data.iloc[:, 2:])
        edge_feats[:, cont_feature_idcs] = edge_scaler.fit_transform(edge_feats)[:, cont_feature_idcs]
        edge_feats = edge_feats.astype(np.float32)
        # Transform edge labels
        edge_labels = edge_labels.astype(np.float32)
        label_scaler = get_composite_transformer_f()
        edge_labels = label_scaler.fit_transform(
            edge_labels.reshape(-1, 1)).reshape(-1)
        num_edge_feats = edge_feats.shape[1]
        return (edge_indices, edge_feats, edge_labels, edge_labels_unscaled,
                label_scaler, edge_scaler, num_edges, num_edge_feats)

    @staticmethod
    def _load_dataset_split(data_base_path):
        data_base_path = pathlib.Path(data_base_path)
        train_idcs = np.load(data_base_path / "train_edge_indices.npy")
        val_idcs = np.load(data_base_path / "val_edge_indices.npy")
        test_idcs = np.load(data_base_path / "test_edge_indices.npy")
        return train_idcs, val_idcs, test_idcs

    @staticmethod
    def _load_bin_data(bin_bounds, edge_labels_unscaled, num_bins,
                       train_idcs, val_idcs, test_idcs):
        # Get edge buckets (assign each edge to a bucket based on magnitude of
        # flow)
        edge_buckets = bin_data(edge_labels_unscaled, num_bins,
                                scale="custom", bin_bounds=bin_bounds)
        # Compute weights for each bucket to counterbalance the imbalanced
        # class/bin distribution
        train_bin_weights = class_weight.compute_class_weight('balanced',
                                                              np.unique(edge_buckets),
                                                              edge_buckets[train_idcs])
        val_bin_weights = class_weight.compute_class_weight('balanced',
                                                              np.unique(edge_buckets),
                                                              edge_buckets[val_idcs])
        test_bin_weights = class_weight.compute_class_weight('balanced',
                                                            np.unique(edge_buckets),
                                                            edge_buckets[test_idcs])
        train_bin_weights = train_bin_weights.astype(np.float32)
        val_bin_weights = val_bin_weights.astype(np.float32)
        test_bin_weights = test_bin_weights.astype(np.float32)
        return edge_buckets, train_bin_weights, val_bin_weights, test_bin_weights

    @staticmethod
    def _load_flow_graph_data(data_base_path, num_nodes, num_edges,
                              flow_adj_threshold):
        # Flow adjacency matrix
        flow_adj_indices = np.load(os.path.join(data_base_path,
                                                "flow_adj_indices.npy")).T
        flow_adj_values = np.load(os.path.join(data_base_path,
                                               "flow_adj_values.npy"))
        flow_adj_matrix = ssp.coo_matrix((flow_adj_values,
                                          (flow_adj_indices[0],
                                           flow_adj_indices[1])),
                                         shape=(num_nodes, num_nodes))
        flow_adj_matrix = flow_adj_matrix.tocsr()

        unweighted_flow_adj_indices = flow_adj_indices[:,
                                      flow_adj_values >= flow_adj_threshold]
        flow_adj_values = flow_adj_values[flow_adj_values >= flow_adj_threshold]
        unweighted_flow_adj_matrix = ssp.coo_matrix(
            (flow_adj_values,
             (unweighted_flow_adj_indices[0], unweighted_flow_adj_indices[1])),
            shape=(num_nodes, num_nodes))
        unweighted_flow_adj_matrix.setdiag(np.ones(num_nodes))
        flow_adj_values = unweighted_flow_adj_matrix.tocoo().data
        flow_adj_indices = np.stack((unweighted_flow_adj_matrix.row,
                                     unweighted_flow_adj_matrix.col), axis=-1)
        flow_adj_indices = flow_adj_indices.astype(np.int64)
        flow_adj_values = flow_adj_values.astype(np.float32)
        unweighted_flow_adj_matrix = (unweighted_flow_adj_matrix > 0.0).astype(np.float)

        # Flow incidence matrix for all edges
        flow_inc_indices = np.load(os.path.join(data_base_path,
                                                "flow_inc_indices.npy"))
        flow_inc_matrix = ssp.coo_matrix(
            (np.ones(flow_inc_indices.shape[1]),
             (flow_inc_indices[0],
              flow_inc_indices[1])),
            shape=(num_nodes, num_edges))
        flow_inc_matrix = flow_inc_matrix.tocsr()

        return (flow_adj_matrix, flow_inc_matrix, flow_adj_indices,
                unweighted_flow_adj_matrix, flow_adj_values)

    @staticmethod
    def _load_geo_graph_data(data_base_path, num_nodes, num_edges,
                             flow_topology):
        # Geographical adjacency matrix
        geo_adj_indices = np.load(os.path.join(data_base_path,
                                               "geo_adj_indices.npy"))
        geo_adj_matrix = ssp.coo_matrix((np.ones(geo_adj_indices.shape[1]),
                                         (geo_adj_indices[0],
                                          geo_adj_indices[1])),
                                        shape=(num_nodes, num_nodes))
        geo_adj_matrix = geo_adj_matrix.tocsr()

        # Geographical incidence matrix for all edges
        geo_inc_indices = np.load(os.path.join(data_base_path,
                                               "geo_inc_indices.npy"))
        geo_inc_matrix = ssp.coo_matrix(
            (np.ones(geo_inc_indices.shape[1]),
             (geo_inc_indices[0],
              geo_inc_indices[1])),
            shape=(num_nodes, num_edges))
        geo_inc_matrix = geo_inc_matrix.tocsr()

        # Get flows for the geographical edges
        all_edges = np.array(flow_topology.adj_matrix.todense()).reshape(-1)            # N^2 matrix
        geo_indices_of_edges = np.array(geo_adj_matrix.todense()).reshape(-1).nonzero() # N^2 matrix
        geo_flows = all_edges[geo_indices_of_edges]
        del all_edges
        all_edges = None
        del geo_indices_of_edges
        geo_indices_of_edges = None
        geo_flows = (geo_flows+1e-5).astype(np.float32)

        return geo_adj_matrix, geo_inc_matrix, geo_adj_indices.T, geo_flows

    @staticmethod
    def _create_data_loaders(train_idcs, val_idcs, test_idcs,
                             train_bin_weights, edge_indices, edge_feats,
                             edge_labels, edge_buckets, batch_size, resample,
                             seed):
        """
        :param train_idcs:
        :param val_idcs:
        :param test_idcs:
        :param train_bin_weights:
        :param edge_indices:
        :param edge_feats:
        :param edge_labels:
        :param edge_buckets:
        :param flow_node_edges_matrix: Transpose of the incidence matrix
        for incoming edges. Shape [E, N].
        :param batch_size:
        :param resample:
        :param seed:
        :return:
        """
        assert (len(edge_indices) == len(edge_feats) == len(edge_labels)
                == len(edge_buckets))

        train_idcs = torch.from_numpy(train_idcs)
        val_idcs = torch.from_numpy(val_idcs)
        test_idcs = torch.from_numpy(test_idcs)

        # Sample weights
        train_sample_weights = train_bin_weights[edge_buckets[train_idcs]]

        # Compute split into training, validation, and test set
        np.random.seed(seed)
        if resample:
            train_sampler = BatchSampler(
                WeightedRandomSampler(train_sample_weights,
                                      train_idcs.shape[0]),
                batch_size=batch_size, drop_last=False)
            train_loader = DataLoader(TensorDataset(edge_indices[train_idcs],
                                                    edge_feats[train_idcs],
                                                    edge_labels[train_idcs],
                                                    edge_buckets[train_idcs]),
                                      batch_sampler=train_sampler)
        else:
            train_loader = DataLoader(TensorDataset(edge_indices[train_idcs],
                                                    edge_feats[train_idcs],
                                                    edge_labels[train_idcs],
                                                    edge_buckets[train_idcs]),
                                      batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(TensorDataset(edge_indices[val_idcs],
                                              edge_feats[val_idcs],
                                              edge_labels[val_idcs],
                                              edge_buckets[val_idcs]),
                                batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(TensorDataset(edge_indices[test_idcs],
                                               edge_feats[test_idcs],
                                               edge_labels[test_idcs],
                                               edge_buckets[test_idcs]),
                                 batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader

    @staticmethod
    def _get_continuous_feature_idcs(df):
        continuous_feature_idcs = []
        for idx, col in enumerate(df.columns):
            if len(df[col].unique()) > 2:
                continuous_feature_idcs.append(idx)
        return continuous_feature_idcs


if __name__ == '__main__':
    ds = UrbanPlanningDataset(data_base_path="Data/London_high/",
                              use_binned_transformer=True,
                              excluded_node_feature_columns=tuple())

    print("\n\nNode features")
    summarize_feature_matrix(ds.node_feats.numpy())
    print("\n\nEdge features")
    summarize_feature_matrix(ds.edge_feats.numpy())