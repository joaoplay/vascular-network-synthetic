import random
from typing import List, Tuple

from networkx import single_source_shortest_path, all_simple_paths

from collections.abc import Callable

import networkx as nx
import numpy as np


""""
IMPORTANT: PLEASE NOTE THAT THIS CODE WAS NOT WRITTEN BY ME. IT WAS WRITTEN BY THE AUTHORS OF THIS GITHUB REPOSITORY:

https://github.com/giodestone/ann-and-pg-city-layout-generator

The code was very useful but needs several improvements in terms of code quality, performance and readability.
I did small changes to make it work with the rest of the code, but not much more. I will try to improve it as much as 
I can in the future. For now, this code is executed when generating the training data (in the preprocessing step) or 
when generating the synthetic graphs. It is not used during training.
"""


def generate_training_samples_for_node(graph: nx.Graph, node_id: int, max_input_paths: int,
                                       max_paths_for_each_reachable_node: int,
                                       max_input_path_length: int, max_output_nodes: int,
                                       distance_function) -> Tuple[List, List]:
    training_sequence = generate_paths_for_node(graph=graph, node_id=node_id,
                                                max_input_paths=max_input_paths,
                                                max_paths_for_each_reachable_node=max_paths_for_each_reachable_node,
                                                max_input_path_length=max_input_path_length,
                                                distance_function=distance_function)

    print("Training sequence length: ", len(training_sequence))

    # training_example = [training_sequence[np.random.randint(len(training_sequence))]]
    training_example = [training_sequence[0]]

    x, y = encode_training_sequence(training_sequence=training_example,
                                    max_input_paths_per_node=max_input_paths,
                                    max_input_path_length=max_input_path_length,
                                    max_output_nodes=max_output_nodes)

    # Squeeze the batch dimension
    x = list(x)
    y = list(y)

    return x, y


def convert_path_to_changes_in_distance(graph: nx.Graph, path: list,
                                        distance_function: Callable[[nx.Graph, int, int], List[float]]) -> list:
    """
    Convert a path to a list of changes in distance between nodes.
    :param graph: The graph the path is in.
    :param path: The path to convert.
    :param distance_function: The distance function to use.
    :return: A list of changes in distance between nodes.
    """

    distances = []

    for i in range(0, len(path) - 1):
        distances.append(distance_function(graph, path[i], path[
            i + 1]))  # If you're getting error here, the incoming path probably doesn't have enough nodes.

    return distances


def get_all_simple_paths_from_node(graph: nx.Graph, node_id, max_input_paths: int,
                                   max_paths_for_each_reachable_node: int, max_input_path_length: int):
    """
    Get all simple paths from a node in a graph.
    :param graph:
    :param node_id:
    :param max_input_paths:
    :param max_paths_for_each_reachable_node:
    :param max_input_path_length:
    :return:
    """
    # Get all nodes reachable from the node_id whose distance is less than max_input_path_length
    all_reachable_nodes = list(single_source_shortest_path(graph, node_id, max_input_path_length))
    # Remove the node_id from the reachable nodes
    all_reachable_nodes = list(filter(lambda a: a != node_id, all_reachable_nodes))
    # Introduce randomness
    random.shuffle(all_reachable_nodes)

    sequence = []  # formatted as [0] = y/predict, [1] = pivot node id, [2] = x/train
    for i in range(len(all_reachable_nodes)):
        # 1. Get paths from the start_node_id to the current reachable node
        paths_from_start_node_id_to_reachable_node = list(all_simple_paths(graph, node_id,
                                                                           all_reachable_nodes[i],
                                                                           max_input_path_length))

        c_max_paths_for_reachable_node = min(max_input_paths - len(sequence), max_paths_for_each_reachable_node)

        # Limit number of paths to max_num_paths_for_each_reachable
        paths_from_start_node_id_to_reachable_node = paths_from_start_node_id_to_reachable_node[
                                                     0:c_max_paths_for_reachable_node]

        for path in paths_from_start_node_id_to_reachable_node:
            # 2. Get all the surrounding nodes
            surrounding_nodes = list(single_source_shortest_path(graph, node_id, 1))

            # Filter out start_node_id
            surrounding_nodes = list(filter(lambda n_id: n_id != node_id, surrounding_nodes))

            # 3.1 remove any paths that cross over surrounding nodes twice.
            if len(set(surrounding_nodes) & set(path)) > 1:
                continue

            # 3. remove surrounding nodes contained in paths
            surrounding_nodes = list(filter(lambda a: a not in path, surrounding_nodes))

            sequence.append((surrounding_nodes, node_id, path))

        if len(sequence) > max_input_paths:
            break

    return sequence


def generate_paths_for_node(graph: nx.graph, node_id: int, max_input_paths: int, max_paths_for_each_reachable_node: int,
                            max_input_path_length: int, distance_function: Callable[[nx.Graph, int, int], List[float]]):
    """
    Generate paths for a node.
    :param graph:
    :param node_id:
    :param max_input_paths:
    :param max_paths_for_each_reachable_node:
    :param max_input_path_length:
    :param distance_function:
    :return:
    """
    # Get "max_num_paths_per_node" paths that can lead to node_id
    paths = get_all_simple_paths_from_node(graph=graph, node_id=node_id, max_input_paths=max_input_paths,
                                           max_paths_for_each_reachable_node=max_paths_for_each_reachable_node,
                                           max_input_path_length=max_input_path_length)

    print("Number of paths: ", len(paths))

    # Find all unique prediction nodes and put them into individual arrays. This must be done as paths
    # predicting separate nodes must be passed in differently.
    unique_nodes = list()
    for path in paths:
        if (path[0], path[1]) not in unique_nodes:
            unique_nodes.append((path[0], path[1]))

    grouped_paths_dict = {}
    for x in paths:
        key = (tuple(x[0]), x[1])
        grouped_paths_dict.setdefault(key, [])
        grouped_paths_dict[key].append(x[2])

    # Ground paths by (surrounding_nodes, start_node_id)
    grouped_paths = [[x for x in paths if x[0] == unique_node[0] and x[1] == unique_node[1]] for unique_node in
                     unique_nodes]

    training_data = []
    for path_group in grouped_paths:
        path_group_training_data = []
        for path in path_group:
            # Surrounding nodes to predict
            prediction_node_ids = path[0]
            # Active node
            current_node_id = path[1]
            # Path
            previous_node_ids = path[2]

            # Calculate relative coordinates between consecutive nodes in the path
            incoming_path_relative_distances = convert_path_to_changes_in_distance(graph=graph, path=previous_node_ids,
                                                                                   distance_function=distance_function)

            # Calculate relative coordinates between the active node and each neighbour
            prediction_nodes_relative_distances = [distance_function(graph, current_node_id, prediction_node_id) for
                                                   prediction_node_id in prediction_node_ids]

            path_group_training_data.append((incoming_path_relative_distances, prediction_nodes_relative_distances))

        training_data.append(path_group_training_data)

    return training_data


def generate_training_data_for_graph(graph: nx.graph, max_input_paths_per_node: int,
                                     max_paths_for_each_reachable_node: int, max_input_path_length: int,
                                     distance_function: Callable[[nx.Graph, int, int], List[float]],
                                     num_iterations: int, max_output_nodes: int):
    """
    Generate a training sequence given a Networkx. For each node in the graph, this method generates max_input_paths_per_node
    paths, each path with a maximum length of max_input_path_length. The max_paths_for_each_reachable_nodes controls the maximum
    number of nodes

    :param graph: The graph to generate the training sequence for.
    :param max_input_paths_per_node: The maximum number of paths to generate for each node in the graph.
    :param max_paths_for_each_reachable_node: The maximum number of paths to generate for each node that is reachable from the current node.
    :param max_input_path_length: The maximum length of each path.
    :param distance_function: The distance function to use when calculating the relative coordinates between nodes.
    :return:
    """
    nodes_to_process = len(graph.nodes())
    all_paths_nodes = []
    for idx in range(num_iterations):
        print(f"Generating training data for iteration {idx + 1} of {num_iterations}")
        for node_id in graph.nodes():
            training_data, training_data_nodes = generate_paths_for_node(graph=graph, node_id=node_id,
                                                                         max_input_paths=max_input_paths_per_node,
                                                                         max_paths_for_each_reachable_node=max_paths_for_each_reachable_node,
                                                                         max_input_path_length=max_input_path_length,
                                                                         distance_function=distance_function)

            # Convert to numpy array with shape (max_input_paths_per_node, max_input_path_length) and initialized to 0
            node_paths_np = np.zeros((len(training_data_nodes), max_input_paths_per_node, max_input_path_length + 1),
                                     dtype=int)
            for i, path_group in enumerate(training_data_nodes):
                padded_group_paths = [path[0] + [0] * (max_input_path_length + 1 - len(path[0])) for path in path_group]
                padded_group_paths = padded_group_paths + [[0] * (max_input_path_length + 1)] * (
                        max_input_paths_per_node - len(padded_group_paths))
                np_paths = np.array(padded_group_paths)
                node_paths_np[i] = np_paths

            all_paths_nodes.append(node_paths_np)

    # Vertically stack all paths
    all_paths_nodes = np.vstack(all_paths_nodes)

    print(all_paths_nodes.shape)

    # Filter out all duplicated paths and return the indices of the unique ones
    unique_paths, unique_indices = np.unique(all_paths_nodes, axis=0, return_index=True)

    print(unique_paths.shape)

    return training_data


def get_signed_distance_between_nodes(graph: nx.Graph, from_node_id, to_node_id):
    """
    Get the signed distance between two nodes as change in x, and y in meters.
    :param graph:
    :param from_node_id:
    :param to_node_id:
    :return:
    """
    current_node = graph.nodes[from_node_id]
    next_node = graph.nodes[to_node_id]

    current_node_pos = np.array(current_node['node_label'])
    next_node_pos = np.array(next_node['node_label'])

    return next_node_pos - current_node_pos


def split_list_into_chunks(l: list, max_size: int):
    """
    Split a list into chunks of maximum length max_size. If the list doesn't evenly divide the last list will be
    smaller than max_size.
    :param l:
    :param max_size:
    :return:
    """
    new_list = []
    max_index = len(l) - 1
    for i in range(0, len(l), max_size):
        if i + max_size > max_index:
            new_list.append(l[i:max_index])
            break
        else:
            new_list.append(l[i:i + max_size])

    return new_list


def encoding_simplified(training_sequence: list, max_input_paths_per_node: int,
                        max_input_path_length: int, max_output_nodes: int):
    # Set up x+y shapes.
    input_shape = (len(training_sequence), max_input_paths_per_node, max_input_path_length, 3)  # x shape
    output_shape = (len(training_sequence), max_input_paths_per_node, max_output_nodes, 3)  # y shape

    # faster than zeroing out anything.
    x = np.zeros(input_shape, dtype='float32')
    y = np.zeros(output_shape, dtype='float32')

    x_length = np.zeros((len(training_sequence), max_input_paths_per_node), dtype='int32')
    y_length = np.zeros((len(training_sequence)), dtype='int32')

    # Encode training sequence.
    train_sequence_index = 0
    for paths in training_sequence:
        path_index = 0
        for path in paths:
            x_length[train_sequence_index, path_index] = len(path[0])
            # Encode input and move the previous path into x
            input_path_node_index = 0
            for input_path_node in path[0]:
                x[train_sequence_index, path_index, input_path_node_index, 0] = input_path_node[0] * 2 + 100 + 1
                x[train_sequence_index, path_index, input_path_node_index, 1] = input_path_node[1] * 2 + 100 + 1
                x[train_sequence_index, path_index, input_path_node_index, 2] = input_path_node[2] * 2 + 100 + 1

                input_path_node_index += 1

            # Encode prediction and set values in y
            prediction_node_index = 0
            for prediction_node in path[1]:
                y[train_sequence_index, path_index, prediction_node_index, 0] = prediction_node[0] * 2 + 100 + 1
                y[train_sequence_index, path_index, prediction_node_index, 1] = prediction_node[1] * 2 + 100 + 1
                y[train_sequence_index, path_index, prediction_node_index, 2] = prediction_node[2] * 2 + 100 + 1

                prediction_node_index += 1

            path_index += 1

        train_sequence_index += 1

    y_length[0] = len(training_sequence[0][0][1])

    return x, y, x_length, y_length


def encode_training_sequence(training_sequence: list, max_input_paths_per_node: int,
                             max_input_path_length: int, max_output_nodes: int):
    # cardinality should be the size of the total range (including negative)
    # up to six incoming nodes, and up to 40 outgoing nodes.

    # suggested output
    # needs to be padded at the start with empty paths [0,0, 0,0, 0,0, 0,0, 0,0, ...] if under
    # [x, y,  x, y,  x, y,  x, y,  x, y,  x, y,  x, y]
    # all x,y need to be zero if they don't exist.

    # suggested output
    # needs to be padded with zeros with paths which have been made empty.
    # [x, y,  x, y,  ...  x, y]
    # all x,y that are empty should be zero.

    no_move = (0.0, 0.0, 0.0)
    padding = (None, None, None)

    empty_incoming_path = []
    for _ in range(max_input_path_length):
        empty_incoming_path.append(no_move)

    empty_prediction = []
    for _ in range(max_output_nodes):
        empty_prediction.append(padding)

    training_sequences_to_remove = []
    training_sequences_to_add = []
    for paths in training_sequence:
        if len(paths) > 5:  # replace 2 with max_num_input_paths
            training_sequences_to_remove.append(paths)
            training_sequences_to_add.extend(split_list_into_chunks(paths, 5))  # replace 2 with max_num_input_paths

    for ts_to_remove in training_sequences_to_remove:
        training_sequence.remove(ts_to_remove)

    training_sequence.extend(training_sequences_to_add)

    training_sequences_to_add.clear()
    training_sequences_to_remove.clear()

    # Set up x+y shapes.
    input_shape = (
        len(training_sequence), max_input_paths_per_node, max_input_path_length, 3)  # x shape
    output_shape = (
        len(training_sequence), max_input_paths_per_node, max_output_nodes, 3)  # y shape

    # faster than zeroing out anything.
    x = np.empty(input_shape, dtype='float32')
    y = np.empty(output_shape, dtype='float32')

    # Encode training sequence.
    # FIXME: Change it to enumerate
    train_sequence_index = 0
    for paths in training_sequence:
        # Pad with empty paths at the start to bring the length up to max_num_input_previous_paths.
        for _ in range(max_input_paths_per_node - len(paths)):
            paths.insert(0, (empty_incoming_path, empty_prediction))

        if len(paths) > max_input_paths_per_node:
            raise Exception(
                "Error: The training sequence has too many paths! The number of paths must be shortened to "
                "max_num_previous_paths.")

        path_index = 0
        # FIXME: Change it to enumerate
        for path in paths:
            # Pad incoming path with empty distances at the start to bring the length up to max_num_input_nodes.
            for _ in range(max_input_path_length - len(path[0])):
                path[0].insert(0, no_move)

            if len(path[0]) > max_input_path_length:
                raise Exception(
                    "Error: The input path is too long! The number of input paths must be shortened to "
                    "max_num_input_nodes.")

            # Pad prediction with empty tuples at the end to bring up the length to max_num_output_nodes.
            if len(path[1]) < max_output_nodes:
                # Insert empty tuple at the end of first prediction.
                path[1].append(no_move)

                for _ in range(max_output_nodes - len(path[1])):
                    path[1].append(padding)

            if len(path[1]) > max_output_nodes:
                raise Exception(
                    "Error: Too many prediction nodes! The number of predictions must be shortened to "
                    "max_num_output_nodes.")

            # Encode input and move the previous path into x
            input_path_node_index = 0
            for input_path_node in path[0]:
                x[train_sequence_index, path_index, input_path_node_index, 0] = input_path_node[0]
                x[train_sequence_index, path_index, input_path_node_index, 1] = input_path_node[1]
                x[train_sequence_index, path_index, input_path_node_index, 2] = input_path_node[2]

                input_path_node_index += 1

            # Encode prediction and set values in y
            prediction_node_index = 0
            for prediction_node in path[1]:
                y[train_sequence_index, path_index, prediction_node_index, 0] = prediction_node[0]
                y[train_sequence_index, path_index, prediction_node_index, 1] = prediction_node[1]
                y[train_sequence_index, path_index, prediction_node_index, 2] = prediction_node[2]

                prediction_node_index += 1

            path_index += 1

        train_sequence_index += 1

    return x, y
