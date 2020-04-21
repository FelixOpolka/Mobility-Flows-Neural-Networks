"""
This module provides functionality for creating a data split of a given input
graph into training, validation and test sets of edges.
"""
import pathlib
import pandas as pd
import numpy as np

from utils import bin_data


def prepare_dataset():
    np.random.seed(7)
    city = "London_high"
    bin_bounds = [10.0, 100.0, 1000.0, 30000.0]
    include_spatial_lag = False
    data_path = pathlib.Path("Data/" + city)

    # Load adjacency matrix specifying which nodes lie in the geographical
    # neighborhood of each other
    geo_adj_matrix, geo_adj_idcs = _load_geo_adj_matrix(city)

    # Load pandas data frames containing node and edge data
    node_data, edge_data = _load_dataframes(city)
    true_flows = edge_data["flows"].values   # also contains 0-valued flows
    edge_idcs = edge_data[["location_1", "location_2"]].values

    num_nodes = geo_adj_matrix.shape[0]
    num_geo_edges = geo_adj_idcs.shape[1]

    print(f"node columns: {node_data.columns}")
    print(node_data.head())
    print(f"edge columns: {edge_data.columns}")
    print(edge_data.head())

    (val_node_idcs, test_node_idcs, val_edge_idcs,
     test_edge_idcs, train_edge_idcs, bin_idcs) = _get_node_split(node_data,
                                                                  edge_data,
                                                                  bin_bounds)

    known_flows = _compute_known_flows(true_flows, edge_idcs, edge_data,
                                       val_node_idcs, test_node_idcs)

    if include_spatial_lag:
        raise NotImplementedError
        # approx_flows, node_data = _substitute_in_approximations(
        #     val_node_idcs, test_node_idcs, flow_adj_idcs, flows,
        #     edge_data["origin_to_neighbourhood"].values,
        #     edge_data["neighbourhood_to_destination"].values, node_data)

    # Drop unused node and edge features
    node_data, edge_data = _filter_feature_data(node_data, edge_data,
                                                include_spatial_lag)

    # Remove 0-valued edges that are not in the training, validation, or test
    # set and update the set indices accordingly
    (train_edge_idcs, val_edge_idcs, test_edge_idcs, edge_idcs, true_flows,
     known_flows, edge_data) = _remove_unused_zero_edges(train_edge_idcs,
                                                         val_edge_idcs,
                                                         test_edge_idcs,
                                                         edge_idcs, true_flows,
                                                         known_flows,
                                                         edge_data)
    flow_adj_idcs, flow_adj_values = _compute_flow_adj_matrix(known_flows, edge_idcs)
    num_flow_edges = flow_adj_idcs.shape[1]

    # Compute incidence graphs from adjacency matrices
    flow_inc_indices = _compute_incidence_matrix(flow_adj_idcs, num_nodes,
                                                 num_flow_edges)
    geo_inc_indices = _compute_incidence_matrix(geo_adj_idcs, num_nodes,
                                                num_geo_edges)

    _store_dataset_files(data_path, edge_data, node_data, true_flows,
                         flow_adj_idcs, flow_adj_values, geo_adj_idcs,
                         flow_inc_indices, geo_inc_indices, train_edge_idcs,
                         val_edge_idcs, test_edge_idcs)


def _load_geo_adj_matrix(city):
    path = f'../raw_data/{city}/geo_adj_matrix.csv'
    geo_adj_matrix = np.genfromtxt(path, delimiter=',')[1:, 1:]
    geo_adj_indices = np.stack(np.nonzero(geo_adj_matrix))
    return geo_adj_matrix, geo_adj_indices


def _load_dataframes(city):
    node_path = f'../raw_data/{city}/node_data.csv'
    node_data = pd.read_csv(node_path, header=0, index_col=0)

    edge_path = f'../raw_data/{city}/edge_data.csv'
    edge_data = pd.read_csv(edge_path, header=0, index_col=0)

    return node_data, edge_data


def _get_node_split(node_data, edge_data, bin_bounds):
    flows = edge_data["flows"].values
    edge_idcs = edge_data.values[:, :2].astype(np.int) # Ex2 array indicating the two nodes an edge connects
    num_edges = len(edge_idcs)
    num_nodes = len(node_data)
    num_bins = len(bin_bounds)
    bin_idcs = bin_data(flows, num_bins, scale="custom", bin_bounds=bin_bounds)
    bin_counts = np.bincount(bin_idcs)
    bin_counts[0] = np.sum((flows > 0) & (flows < bin_bounds[0])) # Special case: When it comes to compute the fraction of edges of the smallest bin, we exclude the huge number of 0-valued edges
    smallest_bin_idx = np.argmin(bin_counts)
    bin_samples, = np.where(bin_idcs == smallest_bin_idx)
    np.random.shuffle(bin_samples)

    test_edge_set, test_node_set = _create_node_set(set(), set(),
                                                    smallest_bin_idx,
                                                    bin_samples, bin_idcs,
                                                    edge_idcs, num_bins,
                                                    0.2 * bin_counts)
    val_edge_set, val_node_set = _create_node_set(test_node_set, test_edge_set,
                                                  smallest_bin_idx,
                                                  bin_samples, bin_idcs,
                                                  edge_idcs, num_bins,
                                                  0.1 * bin_counts)

    # Create training set by selecting all non-zero-valued edges and a limited
    # number of zero-valued edges
    non_train_node_set = val_node_set.union(test_node_set)
    train_edge_set = set()
    max_num_zero = 10000  # include limited number of zero-valued edges
    num_zero = 0
    for edge_idx, (flow, edge_idcs) in enumerate(zip(flows, edge_idcs)):
        if (not edge_idcs[0] in non_train_node_set
                and not edge_idcs[1] in non_train_node_set):
            if flow >= 1.0:
                train_edge_set.add(edge_idx)
            elif num_zero < max_num_zero:
                train_edge_set.add(edge_idx)
                num_zero += 1

    assert len(test_edge_set.intersection(val_edge_set)) == 0
    assert len(test_edge_set.intersection(train_edge_set)) == 0
    assert len(val_edge_set.intersection(train_edge_set)) == 0
    assert len(test_node_set.intersection(val_node_set)) == 0

    val_node_idcs = np.array(list(val_node_set), dtype=np.int)
    test_node_idcs = np.array(list(test_node_set), dtype=np.int)
    val_edge_idcs = np.array(list(val_edge_set), dtype=np.int)
    test_edge_idcs = np.array(list(test_edge_set), dtype=np.int)
    train_edge_idcs = np.array(list(train_edge_set), dtype=np.int)
    return (val_node_idcs, test_node_idcs, val_edge_idcs, test_edge_idcs,
            train_edge_idcs, bin_idcs)


def _create_node_set(unavailable_nodes, unavailable_edges, smallest_bin_idx,
                     bin_samples, bin_idcs, edge_idcs, num_bins, max_per_bin,
                     MAX_PER_BIN_AND_NODE=110):
    """
    We create a validation/test set of edges by randomly drawing edges from
    the bin with the smallest number of samples in them. For each sampled edge,
    we choose one of the incident nodes and add them to a set of nodes excluded
    from training. Then we take the edges incident to that node and add them
    to the validation/test set (except when we already added enough edges for
    a bucket and we also only add a maximum number of edges of the same bucket
    per node; otherwise the most frequent bucket type would contain mostly
    edges of the first few nodes).
    :param unavailable_nodes: Set of node indices that are no longer available
    for being included in the new validation/test set.
    :param unavailable_edges: Set of edge indices that are no longer available
    for being included in the new validation/test set.
    :param smallest_bin_idx: Index of the bin with the fewest samples.
    Determines how to greedily select edges.
    :param bin_samples: NumPy array of edge indices belonging to the smallest
    bin that are used to guide the creation of the node set.
    :param bin_idcs: E-shaped vector specifying for each edge which bin it
    belongs to.
    :param edge_idcs: Ex2-shaped tensor indicating the indices of the two nodes
    an edge connects.
    :param num_bins: Number of bins.
    :param max_per_bin: Maximum number of edges to find for each bin.
    :param MAX_PER_BIN_AND_NODE: Maximum number of edges that a single node
    may add to a single bin.
    :return:
    """
    def add_incident_edges(inc_edge_idcs, current_set_edges,
                           current_set_bin_counts):
        """
        Adds the edges given in `inc_edge_idcs` to `current_set_edges` subject
        to some conditions.
        :param inc_edge_idcs:
        :param current_set_edges:
        :param current_set_bin_counts:
        :return:
        """
        added_count = 0     # Number of edges actually added
        node_bin_counts = np.zeros(num_bins)    # Bin counts for edges actually added
        for inc_edge_idx in inc_edge_idcs:
            edge_bin = bin_idcs[inc_edge_idx]
            if (inc_edge_idx not in current_set_edges
                    and inc_edge_idx not in unavailable_edges
                    and node_bin_counts[edge_bin] < MAX_PER_BIN_AND_NODE
                    and current_set_bin_counts[edge_bin]+node_bin_counts[edge_bin] < max_per_bin[edge_bin]):
                added_count += 1
                node_bin_counts[edge_bin] += 1
                current_set_edges.add(inc_edge_idx)
        return added_count, node_bin_counts

    # Samples in smallest bin
    set_nodes, set_edges = set(), set()     # Nodes and edges selected for validation/test
    set_bin_counts = np.zeros(num_bins)     # Counts of the edges in validation/test set for each bin

    for idx, edge_idx in enumerate(bin_samples):
        # If we have enough edges of the rarest (smallest) type, we can stop
        # adding edges.
        if set_bin_counts[smallest_bin_idx] >= max_per_bin[smallest_bin_idx]:
            break
        # If the edge is already in a different set, do not include it
        if edge_idx in unavailable_edges:
            continue
        out_node, in_node = tuple(edge_idcs[edge_idx])

        # If both nodes are no longer available, go to next edge
        if in_node in unavailable_nodes and out_node in unavailable_nodes:
            continue
        # Decide which of the two nodes to add based on whether one is already
        # in the node set or no longer available
        if in_node in set_nodes or out_node in unavailable_nodes:   # We have already added in_node to set_nodes in a previous iteration OR in_node belongs to an excluded set (e.g. the validation set created in a previous call to this method).
            node_to_add = in_node
        elif out_node in set_nodes or in_node in unavailable_nodes:
            node_to_add = out_node
        else:
            node_to_add = in_node

        # Add node to the set
        set_nodes.add(node_to_add)
        # Now add all the edges incident to the new node to the edge set
        # (subject to some conditions).
        # For outgoing edges
        out_edge_idcs, = np.where(edge_idcs[:, 0] == node_to_add)
        add_set_out_count, add_node_bin_counts = add_incident_edges(out_edge_idcs, set_edges, set_bin_counts)
        set_bin_counts += add_node_bin_counts
        # For incoming edges
        in_edge_idcs,  = np.where(edge_idcs[:, 1] == node_to_add)
        add_set_in_count, add_node_bin_counts = add_incident_edges(in_edge_idcs, set_edges, set_bin_counts)
        set_bin_counts += add_node_bin_counts

        if np.any(set_bin_counts >= max_per_bin):
            print(f"One bin full after adding {idx+1} edges.")
    return set_edges, set_nodes


def _compute_known_flows(true_flows, edge_idcs, edge_data, val_node_idcs,
                         test_node_idcs):
    known_flows = np.copy(true_flows)
    unknown_nodes = np.concatenate((val_node_idcs, test_node_idcs))
    loc1_unknown = np.isin(edge_idcs[:, 0], unknown_nodes)
    loc2_unknown = np.isin(edge_idcs[:, 1], unknown_nodes)
    known_flows[loc1_unknown] = edge_data["neighbourhood_to_location2"].iloc[loc1_unknown]
    known_flows[loc2_unknown] = edge_data["location1_to_neighbourhood"].iloc[loc2_unknown]
    return known_flows


def _substitute_in_approximations(val_nodes, test_nodes, adj_idcs, flows,
                                  o2n_flow_approx, n2d_flow_approx, node_data):
    train_flows = np.copy(flows)

    # Replace flows for edges incident to validation nodes by approximations
    val_outgoing_edge_idcs = np.isin(adj_idcs[0], val_nodes)
    train_flows[val_outgoing_edge_idcs] = n2d_flow_approx[val_outgoing_edge_idcs]
    val_incoming_edge_idcs = np.isin(adj_idcs[1], val_nodes)
    train_flows[val_outgoing_edge_idcs] = o2n_flow_approx[val_incoming_edge_idcs]

    # Replace flows for edges incident to test nodes by approximations
    test_outgoing_edges = np.isin(adj_idcs[0], test_nodes)
    train_flows[test_outgoing_edges] = n2d_flow_approx[test_outgoing_edges]
    test_incoming_edges = np.isin(adj_idcs[1], test_nodes)
    train_flows[test_outgoing_edges] = o2n_flow_approx[test_incoming_edges]

    # Set flows of edge between nodes within the two sets to 0
    union_nodes = np.concatenate((val_nodes, test_nodes))
    inner_edges = np.logical_and((np.isin(adj_idcs[0], union_nodes)), (np.isin(adj_idcs[1], union_nodes)))
    train_flows[inner_edges] = 0.0

    # In node features, replace flow-dependent values of validation/test nodes
    # by their spatial-lag approximation
    node_data.loc[node_data["nodeID"].isin(union_nodes), "out_total"] = node_data["out_total_spatial_lag"]
    node_data.loc[node_data["nodeID"].isin(union_nodes), "in_total"] = node_data["in_total_spatial_lag"]
    node_data.loc[node_data["nodeID"].isin(union_nodes), "gyration_radius"] = node_data["gyration_radius_spatial_lag"]

    return train_flows, node_data


def _remove_unused_zero_edges(train_edge_idcs, val_edge_idcs, test_edge_idcs,
                              edge_idcs, true_flows, known_flows, edge_data):
    """
    We need to change 0-valued edges that are in the training, validation, or
    test set to have a non-zero value because the data set class uses the
    entries of the sparse matrix for some computations and 0-values would just
    be removed, hence messing up the indexing. In particular, we want to only
    specify the edge indices that make up each of these sets. But we cannot
    have 0-valued edges vanish because that would mess up the indexing.
    :param train_edge_idcs:
    :param val_edge_idcs:
    :param test_edge_idcs:
    :param edge_idcs:
    :param true_flows:
    :param known_flows:
    :param edge_data:
    :return:
    """
    # all edges in either the training, validation or test set
    num_edges = len(true_flows)
    tvt_edges = np.concatenate((train_edge_idcs, val_edge_idcs,
                                test_edge_idcs), axis=-1)
    tvt_edges = np.isin(np.arange(num_edges), tvt_edges)   # convert to boolean array

    # Find out the indices of the training, validation, and test edges within
    # the filtered set of edges
    indices = np.zeros(num_edges, dtype=np.int)
    indices[train_edge_idcs] = 1
    indices[val_edge_idcs] = 2
    indices[test_edge_idcs] = 3
    # Remove 0-valued edges that are not in the training, validation, or test
    # set, i.e. keep edges that are in one of the sets or have a non-zero exact
    # or approximate flow.
    retained_edges = tvt_edges | (known_flows != 0.0) | (true_flows != 0.0)
    indices = indices[retained_edges]
    edge_idcs = edge_idcs[retained_edges]
    true_flows = true_flows[retained_edges]
    known_flows = known_flows[retained_edges]
    edge_data = edge_data.iloc[retained_edges]
    # Update indices
    train_edge_idcs = np.where(indices == 1)[0]
    val_edge_idcs = np.where(indices == 2)[0]
    test_edge_idcs = np.where(indices == 3)[0]

    return (train_edge_idcs, val_edge_idcs, test_edge_idcs, edge_idcs,
            true_flows, known_flows, edge_data)


def _compute_flow_adj_matrix(known_flows, edge_idcs):
    """
    :param known_flows: Specifies for each edge the known flow (i.e. true flow
    or spatial lag flow). Shape [E].
    :param edge_idcs: Specifies for each edge the indices of the two incident
    nodes. Shape [E, 2].
    :return:
        - flow_adj_idcs: Indices of non-zero entries in the flow adjacenty
        matrix. Shape [2E, 2].
        - know_flows: Specifies the non-zero values in the flow matrix. Shape
        [2E].
    """
    upper_triag_idcs = edge_idcs[known_flows > 0.0]
    known_flows = known_flows[known_flows > 0.0]
    lower_triag_idcs = np.stack((upper_triag_idcs[:, 1], upper_triag_idcs[:, 0]), axis=1)
    flow_adj_idcs = np.concatenate((upper_triag_idcs, lower_triag_idcs), axis=0)
    flow_adj_values = np.concatenate((known_flows, np.copy(known_flows)))
    return flow_adj_idcs, flow_adj_values


def _compute_incidence_matrix(adj_indices, num_nodes, num_edges):
    # For both incoming and outgoing edges
    inc_matrix = np.zeros((num_nodes, num_edges))
    inc_matrix[adj_indices[0], np.arange(num_edges)] = 1
    inc_matrix[adj_indices[1], np.arange(num_edges)] = 1
    inc_indices = np.stack(np.nonzero(inc_matrix))
    return inc_indices


def _compute_node_idcs_matrix(node_idcs, edge_idcs_set, num_nodes):
    num_edges = len(edge_idcs_set)
    node_idcs_matrix = np.zeros(num_edges, num_nodes)
    node_idcs_matrix[np.arange(num_edges), node_idcs] = 1.0
    node_idcs_matrix[np.arange(num_edges), node_idcs] = 1.0
    return node_idcs_matrix


def _filter_feature_data(node_data, edge_data, include_spatial_lag):
    edge_data = edge_data.drop(["flows"], axis=1)
    node_data = node_data.drop(["nodeID", "in_total_spatial_lag",
                                 "out_total_spatial_lag",
                                 "gyration_radius_spatial_lag"], axis=1)
    if not include_spatial_lag:
        node_data = node_data.drop(["in_total", "out_total", "gyration_rad"],
                                   axis=1)

    return node_data, edge_data


def _store_dataset_files(data_path, edge_data, node_data, flows, flow_adj_idcs,
                         flow_adj_values, geo_adj_idcs, flow_inc_indices,
                         geo_inc_indices, train_edge_idcs, val_edge_idcs,
                         test_edge_idcs):
    data_path.mkdir(exist_ok=True)
    pd.to_pickle(edge_data, data_path / "edge_data.pk")
    pd.to_pickle(node_data, data_path / "node_data.pk")
    np.save(data_path / "flows.npy", flows)
    np.save(data_path / "flow_adj_indices.npy", flow_adj_idcs)
    np.save(data_path / "flow_adj_values.npy", flow_adj_values)
    np.save(data_path / "geo_adj_indices.npy", geo_adj_idcs)
    np.save(data_path / "flow_inc_indices.npy", flow_inc_indices)
    np.save(data_path / "geo_inc_indices.npy", geo_inc_indices)
    np.save(data_path / "train_edge_indices.npy", train_edge_idcs)
    np.save(data_path / "val_edge_indices.npy", val_edge_idcs)
    np.save(data_path / "test_edge_indices.npy", test_edge_idcs)


if __name__ == '__main__':
    prepare_dataset()
