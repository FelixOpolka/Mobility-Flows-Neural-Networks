import math
import numpy as np
import torch
import torch.sparse as sp
import scipy.sparse as ssp


def split_bucketed_data(bin_idcs):
    """
    Splits a given set of samples into the specified number of buckets of
    equal size. Samples are assigned to buckets based on their label. Each
    bucket is split into train, validation, and test set and the overall
    training, validation, and test sets are the concatenation of the individual
    bucket subsets.
    This ensures that train, validation, and test set all contain the same
    number of samples of all sizes.
    :param bin_idcs: Specifies for each label the bucket it belongs to
    :return: Arrays specifying the indices of samples belonging to the
    training, validation, and test set respectively.
    """
    all_train_idcs = []
    all_val_idcs = []
    all_test_idcs = []
    num_bins = torch.max(bin_idcs) + 1
    for idx in range(num_bins):
        bucket_samples, = np.where(bin_idcs == idx)
        np.random.shuffle(bucket_samples)
        split1 = int(0.7 * len(bucket_samples))
        split2 = int(0.8 * len(bucket_samples))
        train_idcs = bucket_samples[:split1]
        val_idcs = bucket_samples[split1:split2]
        test_idcs = bucket_samples[split2:]
        all_train_idcs.append(train_idcs)
        all_val_idcs.append(val_idcs)
        all_test_idcs.append(test_idcs)
    return (np.concatenate(all_train_idcs), np.concatenate(all_val_idcs),
            np.concatenate(all_test_idcs))


def bin_data(labels, num_buckets, scale="linear", base=10, bin_bounds=None):
    """
    Splits the data into specified number of buckets of equal size. Returns for
    each sample the index of the the bucket it belongs to.
    :param labels: Unscaled labels used for computing bucket boundaries and
    assigning samples to buckets.
    :param num_buckets:
    :param scale: Whether to use separate the label domain into buckets on a
    linear or logarithmic scale. Hence the two options are either "linear" or
    "logarithmic".
    :param base: Only relevant if scale="logarithmic". Specifies the base of
    the logarithm.
    :param bin_bounds: Only relevant if scale="custom".
    :return: Array of the same length as labels specifying for each sample
    which bucket it belongs to.
    """
    max_label = np.max(labels)
    if scale == "logarithmic":
        bin_bounds = []
        base_size = max_label / (base**(num_buckets-1))
        for bin_idx in range(num_buckets):
            bin_bounds.append(base**bin_idx * base_size)
        bin_bounds[-1] = bin_bounds[-1] + 1.0
    elif scale == "linear":
        bin_size = int(math.ceil(float(max_label) / float(num_buckets)))
        bin_bounds = [bin_size * idx for idx in range(1, num_buckets+1)]
    elif scale == "custom" and bin_bounds != None:
        if len(bin_bounds) != num_buckets:
            raise ValueError(f"Error: Specified number of bins {num_buckets} "
                             f"does not match specified bin_bounds "
                             f"(length {len(bin_bounds)})")
    else:
        raise ValueError(f"Unknown scale type {scale}")
    print(f"\tBin bounds: {bin_bounds}")
    bin_idcs = np.digitize(labels, bin_bounds)
    return bin_idcs


def to_sparse_tensor(mat):
    """
    Converts a SciPy sparse matrix into a torch sparse tensor.
    """
    if isinstance(mat, ssp.csr_matrix) or isinstance(mat, ssp.csc_matrix):
        mat = mat.tocoo()
    data = mat.data
    indices = np.concatenate((mat.row.reshape(1, -1), mat.col.reshape(1, -1)),
                             axis=0)
    sparse_mat = sp.FloatTensor(torch.LongTensor(indices),
                                torch.FloatTensor(data),
                                torch.Size(mat.shape))
    return sparse_mat


def normalize(mx):
    """
    Row-normalize sparse matrix. Adapted from
    https://github.com/tkipf/pygcn/blob/master/pygcn/utils.py.
    """
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = ssp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def summarize_tensor(x, title=""):
    with torch.no_grad():
        print("-"*10, title, "-"*10, sep="")
        shape = x.shape
        print(f"Shape: {shape}")

        nans = torch.sum(torch.isnan(x))
        print(f"NaNs: {nans}")

        nnz = torch.sum(x < 1e-8)
        print(f"NNZ: {nnz}")

        mean = torch.mean(x)
        print(f"Mean: {mean}")
        std = torch.std(x)
        print(f"Std: {std}")
        median = torch.median(x)
        print(f"Median: {median}")

        min = torch.min(x)
        print(f"Min: {min}")
        max = torch.max(x)
        print(f"Max: {max}")
        print("-"*(20+len(title)))


def summarize_feature_matrix(features):
    for col_idx in range(features.shape[1]):
        values = features[:, col_idx]
        mean = np.mean(values)
        std = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        num_values = len(np.unique(values))
        is_integer = np.sum(np.ceil(values) - values) <= 1e-10
        print("column index:", col_idx)
        print(f"statistics: {mean:.3f} +/- {std:.3f}")
        print(f"min, max: [{min_val:.3f}, {max_val:.3f}]")
        print(f"num unique values: {num_values}")
        print(f"data type:", "integer" if is_integer else "float")
        print()