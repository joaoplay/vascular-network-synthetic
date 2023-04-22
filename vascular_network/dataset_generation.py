import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data

from utils.pyg import convert_to_networkx
from vascular_network.dataset import VesselGraphDataset


def generate_training_graph(dataset_output_path: str, voxel_dim: list = (100.0, 100.0, 100.0),
                            low_degree_threshold: float = 0.01):
    """
    Generate a voxel grid graph from the VesselGraphDataset. The voxel size is defined by the voxel_dim parameter.
    After selecting the voxel, the greatest connected component is extracted and the nodes with a degree lower than
    the low_degree_threshold are removed.
    :param dataset_output_path: Path to save the preprocessed VesselGraph dataset
    :param voxel_dim: The size of the voxel grid
    :param low_degree_threshold: The minimum degree frequency to be considered in the graph
    :return:
    """
    dataset = VesselGraphDataset(root=f'{dataset_output_path}/data', name='BALBc_no1', use_edge_attr=True,
                                 use_atlas=False)
    data = dataset[0].clone()
    data_undirected = Data(x=data.x, edge_index=data.edge_index_undirected, edge_attr=data.edge_attr_undirected)

    c = torch_geometric.nn.voxel_grid(data_undirected.x[:, 0:3], list(voxel_dim))

    # Count unique elements in c tensor
    unique, counts = torch.unique(c, return_counts=True)
    # Get the index of the most common element
    most_common_index = unique[counts.argmax()]

    clustered_data = data_undirected.clone()
    filtered_nodes = torch.argwhere(c == most_common_index).squeeze()
    clustered_data.edge_index = clustered_data.edge_index[:, np.all(np.isin(clustered_data.edge_index, filtered_nodes),
                                                                    axis=0)]

    r_isolated_nodes = torch_geometric.transforms.RemoveIsolatedNodes()
    r_isolated_nodes(clustered_data)

    # Get the degree of each node
    nodes_degree = torch.bincount(clustered_data.edge_index[0])
    # Calculate unique degree frequencies and return counts
    unique_degree_freq, counts = torch.unique(nodes_degree, return_counts=True)
    # Discard degree frequencies that occur less than 5% of the time
    degree_freq = unique_degree_freq[counts > low_degree_threshold * sum(counts)]

    # Filter out nodes whose unique degree frequency is less than 5% of the total number of nodes
    filtered_nodes = torch.argwhere(torch.isin(nodes_degree, degree_freq)).squeeze()
    # Filter out not in filtered nodes
    clustered_data.edge_index = clustered_data.edge_index[:, np.all(np.isin(clustered_data.edge_index, filtered_nodes),
                                                                    axis=0)]

    # Get the largest connected component
    largest_component = torch_geometric.transforms.LargestConnectedComponents()
    r_isolated_nodes(clustered_data)
    largest_component_data = largest_component(clustered_data)
    nx_graph = convert_to_networkx([largest_component_data])[0]

    return nx_graph, largest_component_data


def generate_training_graph_legacy(dataset_output_path: str, voxel_dim: list = (100.0, 100.0, 100.0),
                                   low_degree_threshold: float = 0.01):
    dataset = VesselGraphDataset(root=f'{dataset_output_path}/data', name='BALBc_no1', use_edge_attr=True,
                                 use_atlas=False)
    data = dataset[0].clone()
    data_undirected = Data(x=data.x, edge_index=data.edge_index_undirected, edge_attr=data.edge_attr_undirected)

    c = torch_geometric.nn.voxel_grid(data_undirected.x[:, 0:3], [100.0, 100.0, 100.0])

    # Count unique elements in c tensor
    unique, counts = torch.unique(c, return_counts=True)
    # Get the index of the most common element
    most_common_index = unique[counts.argmax()]

    clustered_data = data_undirected.clone()
    filtered_nodes = torch.argwhere(c == most_common_index).squeeze()
    clustered_data.edge_index = clustered_data.edge_index[:, np.all(np.isin(clustered_data.edge_index, filtered_nodes),
                                                                    axis=0)]

    r_isolated_nodes = torch_geometric.transforms.RemoveIsolatedNodes()
    largest_component = torch_geometric.transforms.LargestConnectedComponents()
    r_isolated_nodes(clustered_data)
    largest_component_data = largest_component(clustered_data)
    nx_graph = convert_to_networkx([largest_component_data])[0]

    return nx_graph, largest_component_data
