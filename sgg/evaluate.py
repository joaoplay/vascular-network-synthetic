import random
from copy import deepcopy
from typing import Any, List

import networkx as nx
import numpy as np
import torch
from matplotlib import pyplot as plt
from nodevectors import GGVec
from sgg.data import generate_training_samples_for_node
from sgg.model import GraphSeq2Seq
from utils.categorical_coordinates_encoder import CategoricalCoordinatesEncoder
from utils.embedding import calculate_embedding_representation
from utils.flow_estimate import annotate_graph_with_flows, compute_radius_from_flow
from utils.radius_class_encoder import RadiusClassEncoder


def random_subgraph(graph, max_depth):
    # Select a random starting node from the graph
    start_node = random.choice(list(graph.nodes()))

    # Generate the ego graph with the specified maximum depth
    subgraph = nx.ego_graph(graph, start_node, radius=max_depth)

    return subgraph


def reset_subgraph_indexes(subgraph):
    mapping = {node: idx for idx, node in enumerate(subgraph.nodes())}
    relabeled_subgraph = nx.relabel_nodes(subgraph, mapping)

    return relabeled_subgraph


def find_main_vessel(graph: nx.Graph):
    """Identify the main vessel of the network: the input node, the output node,
    and the path between them

    The main vessel is defined as the shortest path between the two boundary
    nodes (degree == 1) that are farthest apart

    Every node along the main vessel path gets a 'main_vessel' attribute set
    to True, and every edge along it gets 'main_vessel' = True as well

    Args:
        graph: NetworkX graph

    Returns:
        tuple: (input_node_id, output_node_id, main_vessel_path) where
               main_vessel_path is a list of node ids from input to output
               Returns (None, None, []) when fewer than 2 boundary nodes exist
    """
    boundary_nodes = [n for n in graph.nodes() if graph.degree(n) == 1]

    if len(boundary_nodes) < 2:
        return None, None, []

    #find the pair of boundary nodes with the longest shortest path
    best_pair = None
    best_length = -1
    for i, u in enumerate(boundary_nodes):
        for v in boundary_nodes[i + 1:]:
            try:
                length = nx.shortest_path_length(graph, u, v)
            except nx.NetworkXNoPath:
                continue
            if length > best_length:
                best_length = length
                best_pair = (u, v)

    if best_pair is None:
        return None, None, []

    input_node, output_node = best_pair

    main_vessel_path = nx.shortest_path(graph, input_node, output_node)

    # Mark nodes and edges along the main vessel
    for node in main_vessel_path:
        graph.nodes[node]['main_vessel'] = True
    for u, v in zip(main_vessel_path[:-1], main_vessel_path[1:]):
        graph.edges[u, v]['main_vessel'] = True

    graph.nodes[input_node]['node_type'] = 'input'
    graph.nodes[output_node]['node_type'] = 'output'

    return input_node, output_node, main_vessel_path


def get_starting_map(graph: nx.Graph, depth: int, start_node_id=None):
    """Get a starting map to begin the generation of synthetic graphs.

    After extracting the seed subgraph, identifies the main vessel (input and
    output nodes) and annotates flow and pressure on the graph.

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
    starting_map = reset_subgraph_indexes(starting_map)

    # Determine the unvisited nodes. These are the nodes that have a degree of 1.
    unvisited_nodes = [node_idx for node_idx in starting_map.nodes() if nx.degree(starting_map, node_idx) == 1]
    unvisited_nodes.remove(start_node_id)

    # Identify the main vessel and annotate flows
    find_main_vessel(starting_map)
    annotate_graph_with_flows(starting_map)

    return starting_map, unvisited_nodes


def generate_synthetic_graph(seed_graph: nx.Graph, graph_seq_2_seq: GraphSeq2Seq,
                             categorical_coordinates_encoder: CategoricalCoordinatesEncoder,
                             radius_class_encoder: RadiusClassEncoder | None,
                             unvisited_nodes: list[int], num_iterations: int, max_input_paths: int,
                             max_paths_for_each_reachable_node: int, max_input_path_length: int, max_output_nodes: int,
                             distance_function: callable, max_loop_distance: float, device) -> (nx.Graph, List):
    """
    Generates a synthetic graph using a trained encoder and decoder model. New nodes and edges are added sequentially,
    starting from the seed graph.
    :param seed_graph: Starting graph to generate from.
    :param graph_seq_2_seq: A GraphSeq2Seq trained model.
    :param categorical_coordinates_encoder: Fitted categorical coordinates encoder.
    :param radius_class_encoder: Fitted radius class encoder.
    :param unvisited_nodes: List of unvisited nodes.
    :param num_iterations: Number of iterations to perform.
    :param max_input_paths: Maximum number of input paths to use for each node.
    :param max_paths_for_each_reachable_node: Maximum number of paths to use for each reachable node.
    :param max_input_path_length: Maximum length of each input path.
    :param max_output_nodes: Maximum number of output nodes to generate.
    :param distance_function: Distance function to use for calculating the distance between nodes.
    :param max_loop_distance:
    :param return_steps: Whether to return the intermediate steps of the generation process.
    :param device:
    :return:
    """
    # Copy generated graph from seed graph, so that we don't modify the seed graph.

    generated_graph = seed_graph.copy()

    # Get the greatest node index in the graph to avoid overwriting existing nodes.
    current_node_idx = max(list(generated_graph.nodes())) + 1

    established_loops = 0
    new_nodes = 0
    pending_edges = []

    steps = []

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
        feature_dim = x.shape[-1]
        x_xyz = categorical_coordinates_encoder.transform(x[..., :3])
        if feature_dim > 3 and radius_class_encoder is not None:
            x_radius = radius_class_encoder.transform(x[..., 3])
            nan_mask = torch.isnan(x[..., 3])
            x_radius[nan_mask] = radius_class_encoder.n_classes
            x_encoded = torch.cat([x_xyz, x_radius.unsqueeze(-1)], dim=-1)
        else:
            x_encoded = x_xyz
        x = x_encoded.unsqueeze(0)

        #track edges added during this iteration for flow-based radius adjustment
        new_edges_this_iteration = []

        # Call model to generate new nodes from previously codified paths
        predicted_nodes = graph_seq_2_seq.generate(x)
        for new_node in predicted_nodes:
            # Decode xyz classes with coordinate encoder
            decoded_xyz = categorical_coordinates_encoder.inverse_transform(new_node[:3])

            #decode radius class with radius encoder 
            predicted_radius = None
            predicted_label = None
            if len(new_node) > 3 and radius_class_encoder is not None:
                radius_class = new_node[3].unsqueeze(0)
                #the idea was to decode the radius into label, but if i dont convert to float, it gives error 
                #convert predicted class to float radius 
                predicted_radius = float(radius_class_encoder.class_to_value(radius_class).squeeze(0).item())
                #convert predicted class to label
                predicted_label = radius_class_encoder.inverse_transform(radius_class)
                if isinstance(predicted_label, list):
                    predicted_label = predicted_label[0]

            if torch.any(decoded_xyz):  # Check if xyz coordinates are non-zero
                # Check if the new node is close to an existing node.
                nodes_list = list(generated_graph.nodes)
                # Remove the current node from the list of nodes, so that we don't check if the new node is close to
                # itself.
                current_node_index = nodes_list.index(current_node_id)
                nodes_list.pop(current_node_index)

                # Get the coordinates of the current node.
                current_node_coord = torch.tensor(np.array(generated_graph.nodes[current_node_id]['node_label']),
                                                  device=device)

                # Calculate the coordinates of the new node (xyz only)
                next_node_coord = (current_node_coord + decoded_xyz)

                # Get the coordinates of all the other nodes in the graph.
                current_graph_coordinates = torch.tensor(np.array(list(nx.get_node_attributes(generated_graph,
                                                                                              "node_label").values())),
                                                         device=device)

                # Remove the coordinates of the current node from the list of coordinates.
                start_node_idx = torch.tensor(
                    [i for i in range(current_graph_coordinates.shape[0]) if i != current_node_index], device=device)
                current_graph_coordinates = torch.index_select(current_graph_coordinates, 0, start_node_idx)

                # Calculate the distance between the new node to every other node in the graph (except the current
                # active one)
                dist = torch.nn.functional.pairwise_distance(next_node_coord.unsqueeze(0), current_graph_coordinates)

                if torch.min(dist) <= max_loop_distance:
                    established_loops += 1
                    # If the new node is close to an existing node, add an edge between the current node and the
                    # existing node. A new node is not added.
                    loop_node_index = torch.argmin(dist).item()
                    loop_node_id = list(nodes_list)[loop_node_index]
                    
                    # Add edge with radius attribute if predicted
                    if predicted_radius is not None:
                        generated_graph.add_edge(current_node_id, loop_node_id,
                                                 avgRadiusAvg=predicted_radius,
                                                 avgRadiusLabel=predicted_label)
                    else:
                        generated_graph.add_edge(current_node_id, loop_node_id)

                    new_edges_this_iteration.append((current_node_id, loop_node_id))
                    steps += [(current_node_id, loop_node_id, None)]
                else:
                    new_nodes += 1
                    # Otherwise, add a new node and an edge between the current node and the new node.
                    new_node_id = current_node_idx
                    current_node_idx += 1

                    generated_graph.add_node(new_node_id, node_label=next_node_coord.tolist())
                    
                    # Add edge with radius attribute if predicted
                    if predicted_radius is not None:
                        generated_graph.add_edge(current_node_id, new_node_id,
                                                 avgRadiusAvg=predicted_radius,
                                                 avgRadiusLabel=predicted_label)
                    else:
                        generated_graph.add_edge(current_node_id, new_node_id)
                    
                    new_edges_this_iteration.append((current_node_id, new_node_id))
                    unvisited_nodes.append(new_node_id)

                    steps += [(current_node_id, new_node_id, next_node_coord.tolist())]
            else:
                break

        # Collect new edges for batch flow recomputation
        if new_edges_this_iteration:
            pending_edges.extend(new_edges_this_iteration)

        # Every 100 node expansions, recompute flows and adjust radii in batch
        if pending_edges and (i + 1) % 100 == 0:
            annotate_graph_with_flows(generated_graph)
            for u, v in pending_edges:
                if generated_graph.has_edge(u, v):
                    edge_flow = generated_graph.edges[u, v].get('flow', 0)
                    adjusted_radius = compute_radius_from_flow(edge_flow, u, v, generated_graph)
                    generated_graph.edges[u, v]['avgRadiusAvg'] = adjusted_radius
            pending_edges = []

        if len(unvisited_nodes) == 0:
            # No more unvisited nodes. Stop the generation process.
            break

    #final flow recomputation
    if pending_edges:
        annotate_graph_with_flows(generated_graph)
        for u, v in pending_edges:
            if generated_graph.has_edge(u, v):
                edge_flow = generated_graph.edges[u, v].get('flow', 0)
                adjusted_radius = compute_radius_from_flow(edge_flow, u, v, generated_graph)
                generated_graph.edges[u, v]['avgRadiusAvg'] = adjusted_radius

    #identify the main vessel and annotate flows on the grown graph
    #find_main_vessel(generated_graph)
    generated_graph = annotate_graph_with_flows(generated_graph)

    return generated_graph, steps


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
    # Increase font size of the x and y ticks
    ax[0].tick_params(axis='both', which='major', labelsize=12)
    # Increase title font size
    ax[0].title.set_fontsize(14)
    # Increase tick label font size
    ax[0].xaxis.label.set_fontsize(12)
    ax[0].yaxis.label.set_fontsize(12)

    degree_sequence = sorted((d for n, d in nx_graph.degree()), reverse=True)  # degree sequence
    ax[1].plot(degree_sequence, "b-", marker="o")
    ax[1].set_title("Degree rank plot")
    ax[1].set_ylabel("Degree")
    ax[1].set_xlabel("Rank")
    # Increase font size of the x and y ticks
    ax[1].tick_params(axis='both', which='major', labelsize=12)
    # Increase title font size
    ax[1].title.set_fontsize(14)
    ax[1].xaxis.label.set_fontsize(12)
    ax[1].yaxis.label.set_fontsize(12)

    return fig, ax


def edge_radius_mean_and_std(graph: nx.Graph, default_radius: float = 3.0) -> (float, float):
    """
    compute the average edge radius and standard deviation for edges in the graph.
    :param graph: A networkx graph.
    :param default_radius: Default radius if avgRadiusAvg is not present.
    :return: (mean_radius, std_radius)
    """
    all_radii = []
    for edge in graph.edges:
        radius = graph.edges[edge].get('avgRadiusAvg', default_radius)
        if radius is not None:
            all_radii.append(float(radius))
        else:
            all_radii.append(default_radius)
    
    all_radii = np.array(all_radii)
    mean_radius = np.mean(all_radii) if len(all_radii) > 0 else default_radius
    std_radius = np.std(all_radii) if len(all_radii) > 0 else 0.0
    
    return mean_radius, std_radius


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

    # Calculate the difference of density between the generated graph and the ground truth graph
    generated_graph_density = nx.density(generated_graph)
    ground_truth_graph_density = nx.density(ground_truth_graph)
 
    # Average edge radius of the generated graph
    generated_mean_radius, generated_std_radius = edge_radius_mean_and_std(generated_graph)
    # Average edge radius of the ground truth graph
    ground_truth_mean_radius, ground_truth_std_radius = edge_radius_mean_and_std(ground_truth_graph)

    # Calculate embedding representation of the generated graph
    generated_graph_embed = calculate_embedding_representation(generated_graph)
    # Calculate embedding representation of the ground truth graph
    ground_truth_graph_embed = calculate_embedding_representation(ground_truth_graph)

    # Return dict with evaluation metrics
    return {
        'metrics': {
            'average_clustering_coefficient_difference': generated_graph_avg_clustering_coefficient - ground_truth_graph_avg_clustering_coefficient,
            'average_distance_between_neighbors_difference': generated_mean_distance - ground_truth_mean_distance,
            'standard_deviation_clustering_coefficient_difference': generated_graph_clustering_coefficient_std - ground_truth_graph_clustering_coefficient_std,
            'standard_deviation_distance_between_neighbors_difference': generated_std_distance - ground_truth_std_distance,
            'density_difference': generated_graph_density - ground_truth_graph_density,
            'number_of_nodes_difference': len(generated_graph.nodes) - len(ground_truth_graph.nodes),
            'embedding_distance': np.linalg.norm(generated_graph_embed - ground_truth_graph_embed),
            'average_radius_difference': generated_mean_radius - ground_truth_mean_radius,
            'standard_deviation_radius_difference': generated_std_radius - ground_truth_std_radius,
        },
        'plots': {
            'generated_graph_degree_analysis': generated_graph_degree_analysis[0],
        }
    }
