import sys
import time
import argparse
import os
import warnings
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
import pickle as pk

from torch.nn import Parameter

from layers import DNANodeRepModule, ConvNodeRepModule
from metrics import compute_mae, compute_mape, compute_ssi, compute_geh, \
    compute_cpl, compute_cpc, compute_binned_metric, compute_macro_metric, \
    mae_metric, cpc_metric, cpl_metric, geh_metric, ssi_metric, mape_metric
from dataset import UrbanPlanningDataset
from training_environment import TrainingSettings as ts, PerformanceLogger, NodeConvType, \
    JKType
from training_environment import checkpoint_filepath, OutputLogger

from torch_geometric.nn import JumpingKnowledge


parser = argparse.ArgumentParser(description='UP')
parser.add_argument('--enable-cuda', action='store_true',
                    help='Enable CUDA')
args = parser.parse_args()
args.device = None
if args.enable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')


class EdgeRegressor(nn.Module):

    def __init__(self, num_node_features, num_edge_features, node_rep_size,
                 hidden_dim):
        super(EdgeRegressor, self).__init__()

        # Linear layer to transform target edge features
        self.fc_edges = nn.Sequential(
            nn.Linear(num_edge_features + 2 * num_node_features, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=ts.drop_prob),
            nn.Linear(hidden_dim, hidden_dim),
        )
        concat_hidden_dim = hidden_dim

        if ts.include_node_reps:
            if ts.node_conv_type == NodeConvType.GraphConvolution:
                self.node_rep_module = ConvNodeRepModule(num_node_features,
                                                         node_rep_size,
                                                         ts.num_node_rep_layers,
                                                         ts.improved_gcn,
                                                         ts.drop_prob)
            elif ts.node_conv_type == NodeConvType.DNAConvolution:
                self.node_rep_module = DNANodeRepModule(num_node_features,
                                                        node_rep_size,
                                                        ts.num_node_rep_layers,
                                                        ts.dna_heads,
                                                        ts.dna_groups,
                                                        ts.drop_prob)
            concat_hidden_dim += 2 * node_rep_size

            if ts.jk_type is not JKType.NoJK:
                self.jk = JumpingKnowledge(ts.jk_type.value, channels=8,
                                           num_layers=ts.num_node_rep_layers)
                lin_size = node_rep_size
                if ts.jk_type is JKType.Concat:
                    lin_size = ts.num_node_rep_layers*node_rep_size
                self.jk_lin = nn.Linear(lin_size, node_rep_size)

        self.node_weight = Parameter(torch.from_numpy(np.array(1.0, dtype=np.float32)))
        self.edge_weight = Parameter(torch.from_numpy(np.array(1.0, dtype=np.float32)))

        self.regression_head = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=ts.drop_prob),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=ts.drop_prob),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_nodes, x_edges_batch, edge_indices_batch, edge_indices,
                edge_weight=None):
        """
        :param x_nodes: Node features of shape [N, D]
        :param x_edges_batch: Edge features of shape [B, K]
        :param edge_indices_batch: Matrix of shape [B, 2] indicating the
        indices of the nodes connected by each edge.
        :param edge_indices: Matrix of shape [2, E] indicating for each edge
        in the graph the two node IDs it connects.
        :param edge_weight: Vector of shape [E] containing the edge weight for
        each edge in the graph.
        :return: Predictions for edges with shape [B, 1]
        """

        # Compute hidden representation of target edge
        x_nodes_left = x_nodes[edge_indices_batch[:, 0]]
        x_nodes_right = x_nodes[edge_indices_batch[:, 1]]
        x_concat = torch.cat([x_nodes_left, x_edges_batch, x_nodes_right], dim=-1)
        h_edges = self.fc_edges(x_concat)
        h_total = self.node_weight * h_edges

        # Compute hidden representations of nodes
        if ts.include_node_reps:
            intermediate_node_reps = self.node_rep_module(x_nodes,
                                                          edge_indices.t(),
                                                          edge_weight)
            if ts.jk_type is JKType.NoJK:
                h_nodes = intermediate_node_reps[-1]
            else:
                h_nodes = self.jk(intermediate_node_reps)
                h_nodes = self.jk_lin(h_nodes)
            # Get hidden representations of nodes incident to target edges
            h_nodes_left = h_nodes[edge_indices_batch[:, 0]]
            h_nodes_right = h_nodes[edge_indices_batch[:, 1]]
            h_total += self.edge_weight * h_nodes_left
            h_total += self.edge_weight * h_nodes_right

        regression_output = self.regression_head(h_total)

        return regression_output.squeeze(-1)


def train_epoch(epoch, predictor, data, optimizer, loss_criterion, logger,
                lr_schedule):
    predictor.train()

    for (edge_idcs_batch, x_edges_batch, edge_labels_batch,
         _) in data.train_loader:
        edge_idcs_batch = edge_idcs_batch.to(device=args.device)
        x_edges_batch = x_edges_batch.to(device=args.device)
        edge_labels_batch = edge_labels_batch.to(device=args.device)

        optimizer.zero_grad()
        reg_out = predictor(data.node_feats, x_edges_batch, edge_idcs_batch,
                            data.flow_topology.edge_indices,
                            edge_weight=data.flow_topology.edge_weights)
        loss = loss_criterion(reg_out, edge_labels_batch)
        loss.backward()
        optimizer.step()
        logger.add_values({"train_loss": loss.item()})
    lr_schedule.step()


def validate_epoch(epoch, predictor, data, loss_criterion, data_loader, logger,
                   test):
    predictor.eval()
    prefix = "test" if test else "val"

    for (edge_idcs_batch, x_edges_batch, edge_labels_batch, edge_buckets_batch) in data_loader:
        edge_idcs_batch = edge_idcs_batch.to(device=args.device)
        x_edges_batch = x_edges_batch.to(device=args.device)
        edge_labels_batch = edge_labels_batch.to(device=args.device)

        reg_out = predictor(data.node_feats, x_edges_batch, edge_idcs_batch,
                            data.flow_topology.edge_indices,
                            edge_weight=data.flow_topology.edge_weights)
        loss = loss_criterion(reg_out, edge_labels_batch)
        logger.add_values({
            prefix + "_loss": loss.item(),
            prefix + "_predictions": reg_out.detach().cpu().numpy(),
            prefix + "_labels": edge_labels_batch.detach().cpu().numpy(),
            prefix + "_bins": edge_buckets_batch.detach().cpu().numpy()
        })
    if test:
        with open("preds_labels.pk", "wb") as fd:
            preds = data.label_scaler.inverse_transform(np.concatenate(logger._current_epoch_metrics["test_predictions"], axis=-1).reshape(-1, 1))
            labels = data.label_scaler.inverse_transform(np.concatenate(logger._current_epoch_metrics["test_labels"], axis=-1).reshape(-1, 1))
            pk.dump((preds, labels, logger._current_epoch_metrics["test_node_idcs"]), fd)


def run_training():
    # Set up training environment
    if not os.path.exists(ts.cp_folder):
        os.makedirs(ts.cp_folder)
    log_filepath = checkpoint_filepath(ts.cp_folder, "log", __file__, {},
                                       ".pk")
    summary_filepath = checkpoint_filepath(ts.cp_folder, "summary", __file__,
                                           {}, ".txt")
    output_logger = OutputLogger(checkpoint_filepath(ts.cp_folder, "output",
                                                     __file__, {}, ".txt"))
    sys.stdout = output_logger
    ts.write_summary_file(checkpoint_filepath(ts.cp_folder, "hyperparams",
                                              __file__, {}, "txt"))
    print(ts.settings_description())

    # Load data
    ds = UrbanPlanningDataset(ts.data_base_path, ts.num_bins, ts.batch_size,
                              ts.n_quantiles, ts.resampling,
                              ts.excluded_node_feature_columns,
                              ts.excluded_edge_feature_columns, False,
                              ts.include_edge_flow_feat, ts.adj_flow_threshold,
                              ts.seed)
    # Preprocess data
    ds.to(args.device)

    def _get_metric_funcs(prefix):
        preds_key = prefix+"_predictions"
        labels_key = prefix+"_labels"
        bins_key = prefix+"_bins"
        return {
            prefix+"_loss": (lambda m: np.nanmean(m[prefix+"_loss"])),
            prefix + "_mae": (lambda m: compute_mae(m[preds_key], m[labels_key], ds)),
            prefix + "_binned_mae": (lambda m: compute_binned_metric(mae_metric, m[preds_key], m[labels_key], m[bins_key], ds, ts.num_bins)),
            prefix + "_macro_mae": (lambda m: compute_macro_metric(mae_metric, m[preds_key], m[labels_key], m[bins_key], ds, ts.num_bins)),
            prefix + "_mape": (lambda m: compute_mape(m[preds_key], m[labels_key], ds)),
            prefix + "_binned_mape": (lambda m: compute_binned_metric(mape_metric, m[preds_key], m[labels_key], m[bins_key], ds, ts.num_bins)),
            prefix + "_macro_mape": (lambda m: compute_macro_metric(mape_metric, m[preds_key], m[labels_key], m[bins_key], ds, ts.num_bins)),
            prefix + "_ssi": (lambda m: compute_ssi(m[preds_key], m[labels_key], ds)),
            prefix + "_binned_ssi": (lambda m: compute_binned_metric(ssi_metric, m[preds_key], m[labels_key], m[bins_key], ds, ts.num_bins)),
            prefix + "_macro_ssi": (lambda m: compute_macro_metric(ssi_metric, m[preds_key], m[labels_key], m[bins_key], ds, ts.num_bins)),
            prefix + "_geh": (lambda m: compute_geh(m[preds_key], m[labels_key], ds)),
            prefix + "_binned_geh": (lambda m: compute_binned_metric(geh_metric, m[preds_key], m[labels_key], m[bins_key], ds, ts.num_bins)),
            prefix + "_macro_geh": (lambda m: compute_macro_metric(geh_metric, m[preds_key], m[labels_key], m[bins_key], ds, ts.num_bins)),
            prefix + "_cpl": (lambda m: compute_cpl(m[preds_key], m[labels_key], ds)),
            prefix + "_binned_cpl": (lambda m: compute_binned_metric(cpl_metric, m[preds_key], m[labels_key], m[bins_key], ds, ts.num_bins)),
            prefix + "_macro_cpl": (lambda m: compute_macro_metric(cpl_metric, m[preds_key], m[labels_key], m[bins_key], ds, ts.num_bins)),
            prefix + "_cpc": (lambda m: compute_cpc(m[preds_key], m[labels_key], ds)),
            prefix + "_binned_cpc": (lambda m: compute_binned_metric(cpc_metric, m[preds_key], m[labels_key], m[bins_key], ds, ts.num_bins)),
            prefix + "_macro_cpc": (lambda m: compute_macro_metric(cpc_metric, m[preds_key], m[labels_key], m[bins_key], ds, ts.num_bins)),
        }
    metric_funcs = {
        "train_loss": (lambda m: np.nanmean(m["train_loss"])),
        **_get_metric_funcs("val"),
        **_get_metric_funcs("test"),
    }
    logger = PerformanceLogger(metric_funcs, "val_macro_mae", log_filepath,
                               write_every=ts.write_log_every)

    predictor = EdgeRegressor(ds.num_node_feats, ds.num_edge_feats,
                              hidden_dim=ts.hidden_dim,
                              node_rep_size=ts.node_rep_size)
    predictor = predictor.to(device=args.device)

    optimizer = torch.optim.Adam(predictor.parameters(), lr=ts.lr)
    lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                       list(ts.lr_schedule))
    loss_criterion = (nn.L1Loss() if ts.regression_loss == "L1"
                      else nn.MSELoss())

    print("Start training")
    for epoch in range(-1, ts.num_epochs):
        if epoch >= 0:
            train_epoch(epoch, predictor, ds, optimizer, loss_criterion,
                        logger, lr_schedule)
        validate_epoch(epoch, predictor, ds, loss_criterion, ds.val_loader,
                       logger, test=False)
        validate_epoch(epoch, predictor, ds, loss_criterion, ds.test_loader,
                       logger, test=True)

        logger.complete_epoch()
        print(logger.epoch_summary())
        if epoch % ts.write_log_every == 0:
            logger.write(log_filepath)
    logger.write(log_filepath)
    logger.write_summary(summary_filepath, ts.settings_description())
    return logger


if __name__ == '__main__':
    run_training()
