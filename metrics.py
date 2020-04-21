import numpy as np


def compute_mae(predictions, labels, data):
    preds_unscaled, y_unscaled = _unscale(predictions, labels, data)
    mae = mae_metric(preds_unscaled, y_unscaled)
    return mae


def compute_mape(predictions, labels, data):
    preds_unscaled, y_unscaled = _unscale(predictions, labels, data)
    mape = mape_metric(preds_unscaled, y_unscaled)
    return mape


def compute_ssi(predictions, labels, data):
    preds_unscaled, y_unscaled = _unscale(predictions, labels, data)
    return ssi_metric(preds_unscaled, y_unscaled)


def compute_geh(predictions, labels, data):
    preds_unscaled, y_unscaled = _unscale(predictions, labels, data)
    return geh_metric(preds_unscaled, y_unscaled)


def compute_cpl(predictions, labels, data):
    preds_unscaled, y_unscaled = _unscale(predictions, labels, data)
    return cpl_metric(preds_unscaled, y_unscaled)


def compute_cpc(predictions, labels, data):
    preds_unscaled, y_unscaled = _unscale(predictions, labels, data)
    return cpc_metric(preds_unscaled, y_unscaled)


def compute_binned_metric(metric_f, predictions, labels, bins, data, num_bins):
    bins = np.concatenate(bins, axis=0).reshape(-1)
    preds_unscaled, y_unscaled = _unscale(predictions, labels, data)
    binned_metric = _compute_binned_metric(preds_unscaled, y_unscaled,
                                           bins, num_bins, metric_f)
    return binned_metric


def compute_macro_metric(metric_f, predictions, labels, bins, data, num_bins):
    binned_metric = compute_binned_metric(metric_f, predictions, labels, bins,
                                          data, num_bins)
    macro_metric = (np.nanmean(binned_metric)
                 if not np.all(np.isnan(binned_metric))
                 else np.nan)
    return macro_metric


def _unscale(preds, y, data):
    preds_unscaled = np.concatenate(preds, axis=0).reshape(-1, 1)
    preds_unscaled = data.label_scaler.inverse_transform(preds_unscaled)
    preds_unscaled = preds_unscaled.reshape(-1)
    y_unscaled = np.concatenate(y, axis=0).reshape(-1, 1)
    y_unscaled = data.label_scaler.inverse_transform(y_unscaled)
    y_unscaled = y_unscaled.reshape(-1)
    return preds_unscaled, y_unscaled


def mae_metric(preds_unscaled, y_unscaled):
    mae = np.absolute(preds_unscaled.reshape(-1) - y_unscaled.reshape(-1))
    mae = np.mean(mae)
    return mae


def mape_metric(preds_unscaled, y_unscaled):
    non_zero_target_idcs = y_unscaled > 1e-5
    if np.sum(non_zero_target_idcs) == 0:
        return np.nan
    non_zero_targets = y_unscaled[non_zero_target_idcs]
    predicted = preds_unscaled[non_zero_target_idcs]
    mape = np.absolute(predicted - non_zero_targets) / non_zero_targets
    mape = np.mean(mape, axis=0)
    return mape


def ssi_metric(preds_unscaled, y_unscaled):
    preds_unscaled = preds_unscaled[y_unscaled > 0]
    y_unscaled = y_unscaled[y_unscaled > 0]
    ssi = (np.sum(2 * np.minimum(preds_unscaled, y_unscaled)
                  / (preds_unscaled + y_unscaled))
           / len(y_unscaled))
    return ssi


def geh_metric(preds_unscaled, y_unscaled):
    geh = np.sqrt(2 * (preds_unscaled - y_unscaled)**2
                  / (preds_unscaled + y_unscaled))
    geh_percentage = len(geh[geh < 5]) / len(geh)
    return geh_percentage


def cpl_metric(preds_unscaled, y_unscaled):
    cpl = (2 * np.sum(preds_unscaled * y_unscaled > 1e-8)
           / (np.sum(preds_unscaled > 1e-8) + np.sum(y_unscaled > 1e-8)))
    return cpl


def cpc_metric(preds_unscaled, y_unscaled):
    cpc = (np.sum(2 * np.minimum(preds_unscaled, y_unscaled))
           / (np.sum(preds_unscaled) + np.sum(y_unscaled)))
    return cpc


def _compute_binned_metric(out, y, bins, num_bins, metric_f):
    """
    Computes the given metric for each bin individually.
    :param out: NumPy array containing model predictions.
    :param y: NumPy array containing labels.
    :param bins: NumPy array containing the bins that each label belongs to.
    :param num_bins: Total number of bins.
    :param metric_f: Function which receives the model predictions and labels
    as arguments (in that order) and returns a scalar metric value.
    :return: NumPy array of shape [num_bins] containing the metric value for
    each bin.
    """
    metric_vals = []
    for bin_idx in range(num_bins):
        mask = bins == bin_idx
        if np.sum(mask) > 0:
            vals = metric_f(out[mask], y[mask])
            metric_vals.append(vals)
        else:
            metric_vals.append(np.nan)
    return np.array(metric_vals)
