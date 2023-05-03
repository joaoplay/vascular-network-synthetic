from copy import deepcopy
from typing import Any

import networkx as nx
import numpy as np
import torch
from matplotlib import pyplot as plt

from sgg.data import generate_training_samples_for_node
from sgg.model import GraphSeq2Seq
from utils.categorical_coordinates_encoder import CategoricalCoordinatesEncoder


def get_starting_map(graph: nx.Graph, depth: int, start_node_id=None):
    """Get a starting map to begin the generation of synthetic graphs.

    Args:
        graph (networkx.Graph): The graph with the source nodes used for training.
        depth (int): How many nodes to perform a breadth first search on.
        start_node_id (any, optional): ID of the starting node. Defaults to None.

    Returns:
        networkx.Graph, list: A graph which contains some starting nodes, with cartesian coordinates; the list of unvisited nodes.
    """

    if start_node_id is None:
        start_node_id = min(graph.nodes)

    # Perform a breadth first search on the graph to get a starting map.
    successors = list(nx.bfs_successors(graph, start_node_id, depth_limit=depth))

    nodes = set([suc[0] for suc in successors] + sum([suc[1] for suc in successors], []))

    # Get the partial representation of the graph.
    starting_map = nx.subgraph(graph, nodes)

    # Determine the unvisited nodes. These are the nodes that have a degree of 1.
    unvisited_nodes = [node_idx for node_idx in starting_map.nodes() if nx.degree(starting_map, node_idx) == 1]
    unvisited_nodes.remove(start_node_id)

    return starting_map, unvisited_nodes


def generate_synthetic_graph(seed_graph: nx.Graph, graph_seq_2_seq: GraphSeq2Seq,
                             categorical_coordinates_encoder: CategoricalCoordinatesEncoder,
                             unvisited_nodes: list[int], num_iterations: int, max_input_paths: int,
                             max_paths_for_each_reachable_node: int, max_input_path_length: int, max_output_nodes: int,
                             distance_function: callable, max_loop_distance: float, device) -> nx.Graph:
    """
    Generates a synthetic graph using a trained encoder and decoder model. New nodes and edges are added sequentially,
    starting from the seed graph.
    :param seed_graph: Starting graph to generate from.
    :param graph_seq_2_seq: A GraphSeq2Seq trained model.
    :param categorical_coordinates_encoder: Fitted categorical coordinates encoder.
    :param unvisited_nodes: List of unvisited nodes.
    :param num_iterations: Number of iterations to perform.
    :param max_input_paths: Maximum number of input paths to use for each node.
    :param max_paths_for_each_reachable_node: Maximum number of paths to use for each reachable node.
    :param max_input_path_length: Maximum length of each input path.
    :param max_output_nodes: Maximum number of output nodes to generate.
    :param distance_function: Distance function to use for calculating the distance between nodes.
    :param max_loop_distance:
    :param device:
    :return:
    """

    # Copy generated graph from seed graph, so that we don't modify the seed graph.
    generated_graph = seed_graph.copy()

    # Get the greatest node index in the graph to avoid overwriting existing nodes.
    current_node_idx = max(list(generated_graph.nodes())) + 1

    established_loops = 0
    new_nodes = 0

    for i in range(num_iterations):
        # Pick an unvisited node. This is the node to be expanded.
        current_node_id = unvisited_nodes.pop(0)

        # Perform random walks from the current node and generate the encoded input paths
        x, _ = generate_training_samples_for_node(generated_graph, current_node_id, max_input_paths,
                                                  max_paths_for_each_reachable_node, max_input_path_length,
                                                  max_output_nodes, distance_function)

        # Once x is a list of multiple samples, we need to select one of them randomly.
        # FIXME: Review it! Does it make sense to select a random sample?
        x = x[0]

        # Move to the correct device
        x = torch.Tensor(x).to(device=device)

        # Convert relative coordinates to categorical features
        x = categorical_coordinates_encoder.transform(x).unsqueeze(0)

        # Call model to generate new nodes from previously codified paths
        predicted_nodes = graph_seq_2_seq.generate(x)

        current_node_index = list(generated_graph.nodes).index(current_node_id)
        nodes_to_ignore = [current_node_index]
        for new_node in predicted_nodes:
            # Transform from classes to coordinates
            decoded_new_node = categorical_coordinates_encoder.inverse_transform(new_node)

            if torch.any(decoded_new_node):
                # Check if the new node is close to an existing node.
                nodes_list = list(generated_graph.nodes)
                # Remove the current node from the list of nodes, so that we don't check if the new node is close to
                # itself.
                nodes_list.pop(current_node_index)

                # Get the coordinates of the current node.
                current_node_coord = torch.tensor(np.array(generated_graph.nodes[current_node_id]['node_label']),
                                                  device=device)

                # Calculate the coordinates of the new node.
                next_node_coord = (current_node_coord + decoded_new_node)

                # Get the coordinates of all the other nodes in the graph.
                current_graph_coordinates = torch.tensor(np.array(list(nx.get_node_attributes(generated_graph,
                                                                                              "node_label").values())),
                                                         device=device)

                # Remove the coordinates of the current node from the list of coordinates.
                start_node_idx = torch.tensor(
                    [i for i in range(current_graph_coordinates.shape[0]) if i not in nodes_to_ignore], device=device)
                current_graph_coordinates = torch.index_select(current_graph_coordinates, 0, start_node_idx)

                # Calculate the distance between the new node to every other node in the graph (except the current
                # active one)
                dist = torch.cdist(next_node_coord.unsqueeze(0), current_graph_coordinates)

                if torch.min(dist) <= max_loop_distance:
                    established_loops += 1
                    # If the new node is close to an existing node, add an edge between the current node and the
                    # existing node. A new node is not added.
                    loop_node_index = torch.argmin(dist).item()
                    loop_node_id = list(nodes_list)[loop_node_index]
                    generated_graph.add_edge(current_node_id, loop_node_id)
                else:
                    new_nodes += 1
                    # Otherwise, add a new node and an edge between the current node and the new node.
                    new_node_id = current_node_idx
                    current_node_idx += 1

                    generated_graph.add_node(new_node_id, node_label=next_node_coord.tolist())
                    generated_graph.add_edge(current_node_id, new_node_id)
                    unvisited_nodes.append(new_node_id)
                    nodes_to_ignore.append(new_node_id)
            else:
                break

        if len(unvisited_nodes) == 0:
            # No more unvisited nodes. Stop the generation process.
            break

    print("Established loops: ", established_loops)
    print("New nodes", new_nodes)

    return generated_graph


def edge_length_mean_and_std(graph: nx.Graph) -> (float, float):
    """
    Compute the average edge length of each edge in the graph. The position of each node is the 'node_label' attribute
    of the node. It is a 3D vector.
    :param graph: A networkx graph.
    :return:
    """
    all_distances = []
    for edge in graph.edges:
        all_distances.append(
            np.linalg.norm(np.array(graph.nodes[edge[0]]['node_label']) - np.array(graph.nodes[edge[1]]['node_label'])))

    # Convert to numpy array
    all_distances = np.array(all_distances)
    # Calculate the average distance
    generated_graph_avg_distance_between_neighbors = np.mean(all_distances)
    # Calculate the standard deviation
    generated_graph_std_distance_between_neighbors = np.std(all_distances)

    return generated_graph_avg_distance_between_neighbors, generated_graph_std_distance_between_neighbors


def degree_analysis(nx_graph: nx.Graph):
    """
    Perform degree analysis on the graph. This includes computing the degree distribution and the degree rank plot.
    :param nx_graph: A networkx graph.
    :return:
    """
    # Compute degree distribution of the nx_graph
    nx_graph_degree_distribution = nx.degree_histogram(nx_graph)
    # Draw a side-by-side bar plot of the degree distribution and degree rank plot of the ground truth graph
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].bar(range(len(nx_graph_degree_distribution)), nx_graph_degree_distribution)
    ax[0].set_title('Degree distribution')
    ax[0].set_xlabel('Degree')
    ax[0].set_ylabel('Number of nodes')

    degree_sequence = sorted((d for n, d in nx_graph.degree()), reverse=True)  # degree sequence
    ax[1].plot(degree_sequence, "b-", marker="o")
    ax[1].set_title("Degree rank plot")
    ax[1].set_ylabel("Degree")
    ax[1].set_xlabel("Rank")

    return fig, ax


def compute_graph_comparison_metrics(generated_graph: nx.Graph, ground_truth_graph: nx.Graph) -> dict[str, float | Any]:
    """
    Compute the evaluation metric for the generated graph. Compare the average degree of the generated graph with the
    average degree of the ground truth graph. The metric is the absolute difference between the two values. Also,
    compute the average clustering coefficient of the generated graph and the ground truth graph. The metric is the
    absolute difference between the two values. Finally, compute the average distance between the neighbors of the
    generated graph and the ground truth graph. The position of each node is the 'node_label' attribute of the node. It
    is a 3D vector.
    :param generated_graph: A NetworkX representation of the graph generated by the model.
    :param ground_truth_graph: A NetworkX representation of the ground truth graph.
    :return:
    """

    # Compute degree analysis for the generated graph and the ground truth graph
    generated_graph_degree_analysis = degree_analysis(generated_graph)

    # Compute the average clustering coefficient of the generated graph
    generated_graph_avg_clustering_coefficient = nx.average_clustering(generated_graph)
    # Compute the average clustering coefficient of the ground truth graph
    ground_truth_graph_avg_clustering_coefficient = nx.average_clustering(ground_truth_graph)
    # Compute standard deviation of the clustering coefficient of the generated graph
    generated_graph_clustering_coefficient_std = np.std(list(nx.clustering(generated_graph).values()))
    # Compute standard deviation of the clustering coefficient of the ground truth graph
    ground_truth_graph_clustering_coefficient_std = np.std(list(nx.clustering(ground_truth_graph).values()))

    # Average edge length of the generated graph
    generated_mean_distance, generated_std_distance = edge_length_mean_and_std(generated_graph)
    # Average edge length of the ground truth graph
    ground_truth_mean_distance, ground_truth_std_distance = edge_length_mean_and_std(ground_truth_graph)

    # Return dict with evaluation metrics
    return {
        'metrics': {
            'average_clustering_coefficient_difference': generated_graph_avg_clustering_coefficient - ground_truth_graph_avg_clustering_coefficient,
            'average_distance_between_neighbors_difference': generated_mean_distance - ground_truth_mean_distance,
            'standard_deviation_clustering_coefficient_difference': generated_graph_clustering_coefficient_std - ground_truth_graph_clustering_coefficient_std,
            'standard_deviation_distance_between_neighbors_difference': generated_std_distance - ground_truth_std_distance,
            'number_of_nodes_difference': len(generated_graph.nodes) - len(ground_truth_graph.nodes),
        },
        'plots': {
            'generated_graph_degree_analysis': generated_graph_degree_analysis[0],
        }
    }
