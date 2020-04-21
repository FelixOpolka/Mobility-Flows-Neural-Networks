import sys
import time
from datetime import datetime

import numpy as np
import os
import pickle as pk
from collections import defaultdict
import enum
from pathlib import Path


class NodeConvGraph(enum.Enum):
    Geo = 1
    UnweightedFlow = 2


class NodeConvType(enum.Enum):
    GraphConvolution = 1
    GraphAttention = 2
    GraphNodeEdgeConvolution = 3
    DNAConvolution = 4


class JKType(enum.Enum):
    NoJK = ""
    Concat = "cat"
    MaxPool = "max"
    LSTM = "lstm"


class TrainingSettings:
    ### data set ###
    data_base_path = "Data/London_high"
    resampling = True
    n_quantiles = 1000
    num_bins = 4
    excluded_node_feature_columns = tuple()
    excluded_edge_feature_columns = tuple()

    ### model ###
    hidden_dim = 16             # Dimensionality of any intermediate layers
    edge_rep_size = 16          # Hidden size of the target edge representation
    num_edge_rep_layers = 2     # Number of linear layers for computing the target edge representation
    include_node_reps = True    # Wheather to include node representations at all
    node_rep_size = 16          # Hidden size of the node feature representations
    num_node_rep_layers = 1     # Number of GNN layers for computing node representations
    improved_gcn = False        # Whether to use improved GCN convolutions (i.e. 2 on the adj-matrix diagonal)
    jk_type = JKType.NoJK       # Whether to use JumpingKnowledge skip connections at all and if yet, which type
    node_conv_type = NodeConvType.GraphConvolution
    adj_flow_threshold = 0      # When computing node convolutions based on the flow adjancency matrix, only include edges with a flow greater or equal this threshold
    dna_heads = 1               # Number of attention heads to be used for DNA convolutions
    dna_groups = 1              # Number of channel groups to be used for DNA convolutions
    include_edge_flow_feat = False
    drop_prob = 0.5
    weighted_loss = False
    regression_loss = "L1"      # other option: "L2"

    ### training ###
    cp_folder = "./checkpoints/"
    starting_seed = 7
    seed = 7
    lr = 0.001
    lr_schedule = (50, 65, 80)
    num_epochs = 100
    write_log_every = 20
    batch_size = 64

    if weighted_loss and resampling:
        raise ValueError("Weighted loss and resampling both set to True")

    @staticmethod
    def update_setting(**settings):
        for key, value in settings.items():
            if not hasattr(TrainingSettings, key):
                raise ValueError(f"Attribute {key} not a valid hyperparameter.")
            setattr(TrainingSettings, key, value)
        if TrainingSettings.weighted_loss and TrainingSettings.resampling:
            raise ValueError("Weighted loss and resampling both set to True")

    @staticmethod
    def settigns_dict():
        return {attr: getattr(TrainingSettings, attr)
                for attr in dir(TrainingSettings)
                if (not attr.startswith("__")
                    and not callable(getattr(TrainingSettings,
                                             attr)))}

    @staticmethod
    def settings_description():
        settings_dict = TrainingSettings.settigns_dict()
        return parameter_description_from_dict(settings_dict)

    @staticmethod
    def write_summary_file(filepath):
        filepath = (Path(filepath) if not isinstance(filepath, Path)
                    else filepath)
        settings_dict = TrainingSettings.settigns_dict()
        with filepath.with_suffix(".pk").open("wb") as fd:
            pk.dump(settings_dict, fd)
        settings_description = TrainingSettings.settings_description()
        with filepath.open("w") as fd:
            fd.write(settings_description)


class TrainingEnvironment:

    @staticmethod
    def _process_settings(hyperparam_settings):
        """
        Takes a dictionary of lists/scalars and turns it into a list of
        dictionaries of scalars.
        """
        max_length = max([len(l)
                          for l in hyperparam_settings.values()
                          if type(l) is list]+[1])
        hyperparam_settings = {k: (l if type(l) is list else [l]*max_length)
                               for k, l in hyperparam_settings.items()}
        # Go from dictionary of lists to list of dictionaries
        dicts = [dict(zip(hyperparam_settings, x))
                 for x in zip(*hyperparam_settings.values())]
        return dicts

    @staticmethod
    def hyperparameter_search(training_settings, runs, start_experiment_f):
        """
        :param training_settings: Dictionary containing hyperparameters and
        othr training settings. Keys may only be attributes of the
        TrainingSettings class. Values may be scalars or lists. Using lists
        allows to specify the search space. All lists must have the same
        length.
        :param runs: Number of runs to perform for each training setting.
        :param start_experiment_f: Method for running an experiment. Must
        return a single measure of performance.
        :return:
        """
        dicts = TrainingEnvironment._process_settings(training_settings)
        performances = [defaultdict(list) for _ in range(len(dicts))]
        # Make runs outer-loop to ensure initial results for all settings are
        # obtained asap
        for run_idx in range(runs):
            for settings_idx, settings_dict in enumerate(dicts):
                TrainingSettings.update_setting(**settings_dict)
                TrainingSettings.seed = (TrainingSettings.starting_seed
                                         * 3**run_idx)
                logger = start_experiment_f()
                min_epoch = np.argmin(logger[logger.minimizer])
                performances[settings_idx]["min_epoch"].append(min_epoch)
                for key, values in logger.logs_dict.items():
                    performances[settings_idx][key].append(values[min_epoch])
            # Write results (so far) to file
            TrainingEnvironment.write_summary(dicts, performances)

    @staticmethod
    def write_summary(settings_dicts, performances):
        """
        :param settings_dicts:
        :param performances:
        :return:
        """
        summary = ""
        for settings, performances in zip(settings_dicts, performances):
            description = parameter_description_from_dict(settings)
            summary += description + "\n\n"
            for key in sorted(performances.keys()):
                mean = np.mean(performances[key], axis=0)
                std = np.std(performances[key], axis=0)
                if isinstance(mean, np.ndarray):
                    summary += f"{key}: {mean} +/- {std}\n"
                else:
                    summary += f"{key}: {mean:.5f} +/- {std:.5f}\n"
        if not os.path.exists(TrainingSettings.cp_folder):
            os.makedirs(TrainingSettings.cp_folder)
        with open(os.path.join(TrainingSettings.cp_folder, "summary.txt"), "w") as fd:
            fd.write(summary)


class PerformanceLogger:
    def __init__(self, metric_funcs, minimizer, log_filepath, write_every=20):
        """
        :param metric_funcs: Dictionary of metric functions. A metric function
        accepts exactly one argument which is a dictionary of the metrics from
        a single epoch. It returns the corresponding metric.
        :param minimizer: Key of the variable that determines the final value
        stored in the summary. Usually the validation loss.
        """
        self._metric_funcs = metric_funcs
        self._current_epoch_metrics = defaultdict(list)
        self.logs_dict = defaultdict(list)
        self.minimizer = minimizer
        self.log_filepath = log_filepath
        self.write_every = write_every
        self._write_countdown = write_every
        self._start_time = time.time()
        self._current_epoch = 0

        if "duration" in metric_funcs:
            raise ValueError("Key \"duration\" is a reserved key for internal"
                             "use of PerformanceLogger.")

    def __getitem__(self, key):
        return self.logs_dict[key]

    def __setitem__(self, key, value):
        self.logs_dict[key] = value

    def add_values(self, metric_dict):
        for key, metric_batch in metric_dict.items():
            self._current_epoch_metrics[key].append(metric_batch)
        # Write log if necessary
        self._write_countdown -= 1
        if self._write_countdown == 0:
            self.write(self.log_filepath)
            self._write_countdown = self.write_every

    def complete_epoch(self):
        """
        Marks the epoch as finished and computes the epoch's metrics.
        """
        for key, metric_func in self._metric_funcs.items():
            metric = metric_func(self._current_epoch_metrics)
            self.logs_dict[key].append(metric)
        # add duration as additional metric
        duration = time.time() - self._start_time
        self.logs_dict["duration"].append(duration)
        self._start_time = time.time()

        self._current_epoch_metrics = defaultdict(list)
        self._current_epoch += 1

    def epoch_summary(self):
        summary = f"{self._current_epoch}:"
        for key, vals in self.logs_dict.items():
            if key == "duration":
                continue
            if isinstance(vals[-1], np.ndarray):
                summary += f"\t{key}: {vals[-1]}\n"
            else:
                summary += f"\t{key}: {vals[-1]:.5f}\n"
        duration = self.logs_dict["duration"][-1]
        summary += f"\t[{duration:.2f}s]"
        return summary

    def min(self, key):
        if key not in self.logs_dict or len(self.logs_dict[key]) == 0:
            return 10e8     # Find nicer way
        return min(self.logs_dict[key])

    def write(self, filepath):
        with open(filepath, "wb") as fd:
            pk.dump(self.logs_dict, fd)

    def write_summary(self, filepath, settings_description=""):
        """
        Writes a text file containing a summary of the run.
        :param filepath: Filepath of the summary.
        """
        min_idx = np.argmin(self.logs_dict[self.minimizer])
        summary = settings_description + "\n\n"
        for key, value in self.logs_dict.items():
            min_val = value[min_idx]
            summary += f"{key}: {min_val}\n"
        with open(filepath, "w") as fd:
            fd.write(summary)


class OutputLogger:
    """
    Overrides normal stdout, i.e. `sys.stdout = OutputLogger(...)`. After that,
    print writes all output to stdout *and* the specified log-file.
    """
    def __init__(self, log_filepath):
        self.terminal = sys.stdout
        self.log = open(log_filepath, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.flush()
        self.log.close()


def parameter_description_from_dict(dict):
    return dict.__str__()[1:-1].replace("\'", "").replace(": ", "=")


def checkpoint_filepath(directory, basename, script_file, parameters,
                        file_ending, add_timestamp=True):
    """
    Automatically creates a checkpoint filepath with the given attributes.
    :param directory: Directory in which the file should be stored.
    :param basename: Base name of the file, which will be appended by some
    extra parameters (see below).
    :param script_file: Name of the script as returned by __file__.
    :param parameters: Dictionary of additional parameters.
    :param file_ending: File ending of the file.
    :return: Filepath as a string.
    """
    script_name = os.path.basename(script_file)
    script_name = script_name[:script_name.find(".")]

    param_descr = parameter_description_from_dict(parameters)

    if file_ending[0] != ".":
        file_ending = "." + file_ending

    if add_timestamp:
        time_stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename = (script_name + "," + param_descr + "_"
                    + time_stamp + "_" + basename + file_ending)
    else:
        filename = script_name + "," + param_descr + "_" + basename + file_ending

    filepath = os.path.join(directory, filename)
    return filepath


if __name__ == '__main__':
    TrainingSettings.update_setting(**{"lr": 0.1})
