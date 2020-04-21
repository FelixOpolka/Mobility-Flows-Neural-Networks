import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sp
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import GCNConv, DNAConv


def tensor_normalize(matrix):
    row_sum = sp.sum(matrix, dim=1).to_dense()
    # if torch.any(row_sum == 0.0):
    #     raise ValueError("Matrix contains rows with sum 0.")
    r_inv = row_sum.pow(-1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.0
    norm_matrix = torch.matmul(torch.diag(r_inv), matrix.to_dense())
    return norm_matrix


class ConvNodeRepModule(Module):
    def __init__(self, in_dim, hidden_dim, num_layers, improved_gcn,
                 drop_prob):
        super(ConvNodeRepModule, self).__init__()
        self.conv_layers = []
        for idx in range(num_layers):
            cur_in_dim = in_dim if idx == 0 else hidden_dim
            cur_layer = NormalizedRegularizedGCNLayer(cur_in_dim, hidden_dim,
                                                      improved_gcn, drop_prob)
            self.conv_layers.append(cur_layer)
        self.conv_layers = nn.ModuleList(self.conv_layers)

    def forward(self, node_features, edge_indices, edge_weight=None):
        h = node_features
        intermediate_reps = []
        for layer in self.conv_layers:
            h = layer(h, edge_indices, edge_weight)
            intermediate_reps.append(h)
        return intermediate_reps


class NormalizedRegularizedGCNLayer(Module):
    def __init__(self, in_dim, out_dim, improved_gcn, drop_prob):
        super(NormalizedRegularizedGCNLayer, self).__init__()
        self.gcn = GCNConv(in_dim, out_dim, improved_gcn)
        self.bn = nn.BatchNorm1d(out_dim)
        self.drop = nn.Dropout(drop_prob)

    def forward(self, node_features, edge_indices, edge_weight=None):
        h = self.gcn(node_features, edge_indices, edge_weight)
        h = F.relu(h)
        h = self.bn(h)
        h = self.drop(h)
        return h


class DNANodeRepModule(Module):
    """
    Applies a given number of DNA convolutions on the given data. Returns a
    list of all the intermediate representations after each layer.
    """

    def __init__(self, in_dim, hidden_dim, num_layers, dna_heads, dna_groups,
                 drop_prob):
        super(DNANodeRepModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.pre_lin = nn.Linear(in_dim, hidden_dim)
        self.pre_drop = nn.Dropout(drop_prob)
        self.conv_layers = []
        for _ in range(num_layers):
            cur_layer = NormalizedRegularizedDNALayer(hidden_dim, dna_heads,
                                                      dna_groups, drop_prob)
            self.conv_layers.append(cur_layer)
        self.conv_layers = nn.ModuleList(self.conv_layers)

    def forward(self, input, edge_indices):
        h = F.relu(self.pre_lin(input))
        h = self.pre_drop(h)
        h = h.view(-1, 1, self.hidden_dim)
        intermediate_reps = []
        for conv in self.conv_layers:
            h_new = conv(h, edge_indices)
            intermediate_reps.append(h_new)
            h_new = h_new.view(-1, 1, self.hidden_dim)
            h = torch.cat([h, h_new], dim=1)
        return intermediate_reps


class NormalizedRegularizedDNALayer(Module):
    def __init__(self, channels, heads, groups, dropout):
        super(NormalizedRegularizedDNALayer, self).__init__()
        self.dna = DNAConv(channels, heads, groups, dropout)
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, all_node_features, edge_indices):
        h = self.dna(all_node_features, edge_indices)
        h = F.relu(h)
        h = self.bn(h)
        return h


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    Adapted from https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py
    """

    def __init__(self, in_features, out_features, adj_matrix, edge_feat_matrix,
                 attention_scores=None, bias=True):
        """
        :param in_features:
        :param out_features:
        :param bias:
        :param attention_scores: Sparse tensor containing for each pair of
        nodes the attention score between the nodes. Shape [N, N].
        :param adj_matrix:
        """
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.adj_matrix = tensor_normalize(adj_matrix)
        # Pre-compute attention adjacency matrix
        if attention_scores is not None:
            with torch.no_grad():
                self.adj_matrix = tensor_normalize(adj_matrix * attention_scores)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        """
        :param input: Node features of shape [N, D]
        """
        support = torch.mm(input, self.weight)  # Shape [N, K]
        output = torch.matmul(self.adj_matrix, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphNodeEdgeConvolution(Module):

    def __init__(self, node_input_size, edge_input_size, output_size,
                 adj_matrix, edge_feat_matrix, bias=True):
        """
        :param node_input_size:
        :param edge_input_size:
        :param output_size:
        :param adj_matrix:
        :param edge_feat_matrix: [N, N, K]
        :param bias:
        """
        super(GraphNodeEdgeConvolution, self).__init__()
        self.input_size = node_input_size + edge_input_size
        self.output_size = output_size
        self.weight = Parameter(torch.FloatTensor(self.input_size, output_size))
        if bias:
            self.bias = Parameter(torch.FloatTensor(output_size))
        else:
            self.register_parameter('bias', None)
        self.adj_matrix = tensor_normalize(adj_matrix)
        self.edge_feat_matrix = edge_feat_matrix
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, node_features):
        num_nodes = node_features.shape[0]
        node_feats = node_features.view(1, num_nodes, -1).expand(num_nodes, -1, -1)
        combined_feats = torch.cat([self.edge_feat_matrix, node_feats], dim=-1)
        support = torch.matmul(combined_feats, self.weight)  # shape [N, N, D]
        output = torch.matmul(support.transpose(2, 0), self.adj_matrix)     # shape [D, N, N]
        output = torch.diagonal(output, dim1=1, dim2=2)
        output = output.transpose(1, 0)
        return output


class EdgeConvolution(nn.Module):
    def __init__(self, in_features, out_features, inc_matrix):
        """
        :param in_features:
        :param out_features:
        :param inc_matrix: Sparse incidence matrix of the graph of shape
        [N, E].
        """
        super(EdgeConvolution, self).__init__()
        self.weight = nn.parameter.Parameter(torch.FloatTensor(in_features,
                                                               out_features))
        self.bias = nn.parameter.Parameter(torch.FloatTensor(out_features))
        self.inc_matrix = inc_matrix
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, edge_nodes, edge_feats):
        """
        :param edge_nodes: Matrix indicating the nodes which each edge in the
        batch connects. Shape [B, N].
        :param edge_feats: Features of *all* edges in the graph. Shape [E, D].
        :return: Hidden representation of shape [B, K].
        """
        # Get edges incident to the left and right nodes of each edge in the
        # batch. Result has shape [B, E].
        batch_edge_idcs = sp.mm(self.inc_matrix.transpose(1, 0),
                                edge_nodes.transpose(1, 0)).transpose(1, 0)
        # Normalise matrix row-wise such that edge features are averaged, not
        # summed.
        row_sum = torch.sum(batch_edge_idcs, dim=1)
        inv = 1.0 / row_sum
        inv[torch.isinf(inv)] = 0.0
        batch_edge_idcs = batch_edge_idcs * inv.view(-1, 1)

        # Compute hidden representations from edge_features
        h_edges = torch.mm(edge_feats, self.weight) + self.bias     # [E, K]

        # Obtain features of each of these edges
        h = torch.spmm(batch_edge_idcs, h_edges)  # [B, K]

        return h


class DeepGraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    Adapted from https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py
    """

    def __init__(self, in_features, hidden_features, out_features, adj_matrix,
                 attention_scores=None, bias=True, num_layers=1):
        """
        :param in_features:
        :param out_features:
        :param bias:
        :param attention_scores: Sparse tensor containing for each pair of
        nodes the attention score between the nodes. Shape [N, N].
        :param adj_matrix:
        """
        super(DeepGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lin_layers = []
        self.bns = []
        for idx in range(num_layers):
            in_size = in_features if idx == 0 else hidden_features
            out_size = out_features if idx == num_layers-1 else hidden_features
            self.lin_layers.append(nn.Linear(in_size, out_size, bias=bias))
            self.bns.append(nn.BatchNorm1d(out_size))
        self.lin_layers = nn.ModuleList(self.lin_layers)
        self.bns = nn.ModuleList(self.bns)
        # Pre-compute adjacency matrix with weighting if necessary
        self.adj_matrix = adj_matrix
        if attention_scores is not None:
            self.adj_matrix = tensor_normalize(adj_matrix * attention_scores)

    def forward(self, input):
        support = input
        for idx in range(len(self.lin_layers)):
            support = self.lin_layers[idx](support)
            if idx < len(self.lin_layers)-1:
                support = F.relu(support)
                support = self.bns[idx](support)
        output = torch.matmul(self.adj_matrix, support)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class DeepEdgeConvolution(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, inc_matrix,
                 bias=True, num_layers=1):
        """
        :param in_features:
        :param out_features:
        :param inc_matrix: Sparse incidence matrix of the graph of shape
        [N, E].
        """
        super(DeepEdgeConvolution, self).__init__()
        self.lin_layers = []
        self.bns = []
        for idx in range(num_layers):
            in_size = in_features if idx == 0 else hidden_features
            out_size = out_features if idx == num_layers-1 else hidden_features
            self.lin_layers.append(nn.Linear(in_size, out_size, bias=bias))
            self.bns.append(nn.BatchNorm1d(out_size))
        self.lin_layers = nn.ModuleList(self.lin_layers)
        self.bns = nn.ModuleList(self.bns)
        self.inc_matrix = inc_matrix

    def forward(self, edge_nodes, edge_feats):
        """
        :param edge_nodes: Matrix indicating the nodes which each edge in the
        batch connects. Shape [B, N].
        :param edge_feats: Features of *all* edges in the graph. Shape [E, D].
        :return: Hidden representation of shape [B, K].
        """
        # Get edges incident to the left and right nodes of each edge in the
        # batch. Result has shape [B, E].
        batch_edge_idcs = sp.mm(self.inc_matrix.transpose(1, 0),
                                edge_nodes.transpose(1, 0)).transpose(1, 0)
        # Normalise matrix row-wise such that edge features are averaged, not
        # summed.
        row_sum = torch.sum(batch_edge_idcs, dim=1)
        inv = 1.0 / row_sum
        inv[torch.isinf(inv)] = 0.0
        batch_edge_idcs = batch_edge_idcs * inv.view(-1, 1)

        # Compute hidden representations from edge_features
        h_edges = edge_feats
        for idx in range(len(self.lin_layers)):
            h_edges = self.lin_layers[h_edges]
            if idx < len(self.lin_layers)-1:
                h_edges = F.relu(h_edges)
                h_edges = self.bns[idx](h_edges)

        # Obtain features of each of these edges
        h = torch.spmm(batch_edge_idcs, h_edges)  # [B, K]

        return h


class PatchToPatchEdgeConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(PatchToPatchEdgeConvolution, self).__init__()
        self.weight = nn.parameter.Parameter(torch.FloatTensor(in_features,
                                                               out_features))
        self.bias = nn.parameter.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, edge_nodes, adj_matrix, inc_matrix, edge_feats):
        """
        :param edge_nodes: Matrix indicating the nodes which each edge in the
        batch connects. Shape [B, N]
        :param adj_matrix: Sparse adjacency matrix of the graph of shape
        [N, N]. Must contain only 1-entries (i.e. should not be normalised).
        :param inc_matrix: Sparse incidence matrix of the graph of shape
        [N, E].
        :param edge_feats: Features of *all* edges in the graph. Shape [E, D].
        :return: Hidden representation of shape [B, K].
        """
        # Get edges incident to the left and right nodes of each edge in the
        # batch. Result has shape [B, E].
        # In essence, it computes BxN * NxN * NxE
        # = edge_nodes * adj_matrix * inc_matrix.
        batch_edge_idcs = sp.mm(adj_matrix.transpose(1, 0),
                                edge_nodes.transpose(1, 0))
        batch_edge_idcs = sp.mm(inc_matrix.transpose(1, 0),
                                batch_edge_idcs).transpose(1, 0)
        # Find exactly those edges which are two "hops" away from the edge
        # in the batch
        batch_edge_idcs = (batch_edge_idcs == 2.0).float()
        # Normalise matrix row-wise such that edge features are averaged, not
        # summed.
        row_sum = torch.sum(batch_edge_idcs, dim=1)
        inv = 1.0 / row_sum
        inv[torch.isinf(inv)] = 0.0
        batch_edge_idcs = batch_edge_idcs * inv.view(-1, 1)

        # Compute hidden representations from edge_features
        h_edges = torch.mm(edge_feats, self.weight) + self.bias     # [E, K]

        # Obtain features of each of these edges
        h = torch.spmm(batch_edge_idcs, h_edges)  # [B, K]

        return h


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, adj_matrix, dropout, alpha,
                 edge_feats, edge_idcs, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.adj_matrix = adj_matrix
        self.alpha = alpha
        self.concat = concat
        num_nodes = adj_matrix.shape[0]
        self.edge_feats = torch.zeros(
            (num_nodes, num_nodes, edge_feats.shape[1])).to(
            device=edge_idcs.device)
        self.edge_feats[edge_idcs[:, 0], edge_idcs[:, 1], :] = edge_feats

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(
            size=(2 * out_features + self.edge_feats.shape[-1], 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        a_input = torch.cat([a_input, self.edge_feats], dim=-1)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(self.adj_matrix > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

