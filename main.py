from regression_model import run_training
from training_environment import TrainingEnvironment, NodeConvGraph, NodeConvType, JKType

if __name__ == '__main__':
    hyperparameters = {
        # data set
        "data_base_path": "Data/London_high",
        "resampling": True,
        "n_quantiles": 5000,
        "num_bins": 4,
        "excluded_node_feature_columns": tuple(),
        "excluded_edge_feature_columns": tuple(),
        # model
        "hidden_dim": 16,
        "edge_rep_size": 8,
        "num_edge_rep_layers": 2,
        "include_node_reps": True,
        "node_rep_size": 8,
        "num_node_rep_layers": 1,
        "improved_gcn": True,
        "jk_type": JKType.NoJK,
        "node_conv_type": NodeConvType.GraphConvolution,
        "adj_flow_threshold": 0,
        "dna_heads": 1,
        "dna_groups": 1,
        "include_edge_flow_feat": False,
        "drop_prob": 0.3,
        "weighted_loss": False,
        "regression_loss": "L2",
        # training
        "cp_folder": "./checkpoints/test",
        "lr": 0.01,
        "lr_schedule": (50, 65, 80, 95),
        "num_epochs": 110,
        "batch_size": 256,
    }
    TrainingEnvironment.hyperparameter_search(hyperparameters, 3, run_training)
