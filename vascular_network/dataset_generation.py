import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data

from utils.pyg import convert_to_networkx
from vascular_network.dataset import VesselGraphDataset


def generate_training_graph(dataset_output_path: str, voxel_dim: list = (1000.0, 1000.0, 1000.0),
                            low_degree_threshold: float = 0.01, voxel_index: int = None):
    """
    Generate a voxel grid graph from the VesselGraphDataset. The voxel size is defined by the voxel_dim parameter.
    After selecting the voxel, the greatest connected component is extracted and the nodes with a degree lower than
    the low_degree_threshold are removed.
    :param dataset_output_path: Path to save the preprocessed VesselGraph dataset
    :param voxel_dim: The size of the voxel grid
    :param low_degree_threshold: The minimum degree frequency to be considered in the graph
    :param voxel_index: If provided, select this specific voxel index instead of the most populated one
    :return:
    """
    dataset = VesselGraphDataset(root=f'{dataset_output_path}/data', name='synthetic_graph_1', use_edge_attr=True,
                                 use_atlas=False)
    data = dataset[0].clone()
    data_undirected = Data(x=data.x, edge_index=data.edge_index_undirected, edge_attr=data.edge_attr_undirected)

    c = torch_geometric.nn.voxel_grid(data_undirected.x[:, 0:3], list(voxel_dim))

    # Count unique elements in c tensor
    unique, counts = torch.unique(c, return_counts=True)

    if voxel_index is not None:
        selected_voxel = voxel_index
    else:
        selected_voxel = unique[counts.argmax()].item()

    clustered_data = data_undirected.clone()
    filtered_nodes = torch.argwhere(c == selected_voxel).squeeze()
    #use torch operations to compute a boolean mask of edges whose both endpoints are in filtered_nodes
    edge_index_full = clustered_data.edge_index
    mask = torch.all(torch.isin(edge_index_full, filtered_nodes), dim=0)
    clustered_data.edge_index = edge_index_full[:, mask]
    clustered_data.edge_attr = clustered_data.edge_attr[mask]

    r_isolated_nodes = torch_geometric.transforms.RemoveIsolatedNodes()
    r_isolated_nodes(clustered_data)

    
    # Get the degree of each node
    nodes_degree = torch.bincount(clustered_data.edge_index[0])
    # Calculate unique degree frequencies and return counts, excluding degree-0 (isolated) nodes
    nonzero_mask = nodes_degree > 0
    nonzero_degrees = nodes_degree[nonzero_mask]
    unique_degree_freq, counts = torch.unique(nonzero_degrees, return_counts=True)
    # Discard degree frequencies that occur less than 5% of the time
    degree_freq = unique_degree_freq[counts > low_degree_threshold * sum(counts)]


    # Filter out nodes whose unique degree frequency is less than 5% of the total number of nodes
    filtered_nodes = torch.argwhere(torch.isin(nodes_degree, degree_freq)).squeeze()
    # Filter out not in filtered nodes
    edge_index_full = clustered_data.edge_index
    mask = torch.all(torch.isin(edge_index_full, filtered_nodes), dim=0)
    clustered_data.edge_index = edge_index_full[:, mask]
    clustered_data.edge_attr = clustered_data.edge_attr[mask]


    # Get the largest connected component
    largest_component = torch_geometric.transforms.LargestConnectedComponents()
    largest_component_data = largest_component(clustered_data)
    nx_graph = convert_to_networkx([largest_component_data])[0]

    edge_index = largest_component_data.edge_index
    edge_attr = largest_component_data.edge_attr

    for i in range(edge_index.shape[1]):
        u = int(edge_index[0, i])
        v = int(edge_index[1, i])
        r = float(edge_attr[i, 2]) #radius is the 3rd attribute in edge_attr 
        if nx_graph.has_edge(u, v):
            nx_graph[u][v]["avgRadiusAvg"] = r
    return nx_graph, largest_component_data


def get_all_voxel_indices(dataset_output_path: str, voxel_dim: list = (100.0, 100.0, 100.0)):
    """
    Get a list of all voxel indices
    :param dataset_output_path: Path to the preprocessed VesselGraph dataset
    :param voxel_dim: The size of the voxel grid
    :return: List of (voxel_index, node_count) tuples, sorted by voxel index 
    """
    dataset = VesselGraphDataset(root=f'{dataset_output_path}/data', name='synthetic_graph_1', use_edge_attr=True,
                                 use_atlas=False)
    data = dataset[0].clone()
    data_undirected = Data(x=data.x, edge_index=data.edge_index_undirected, edge_attr=data.edge_attr_undirected)

    c = torch_geometric.nn.voxel_grid(data_undirected.x[:, 0:3], list(voxel_dim))
    unique, counts = torch.unique(c, return_counts=True)

    voxels = [(int(idx.item()), int(cnt.item())) for idx, cnt in zip(unique, counts)]
    voxels.sort(key=lambda x: x[0])
    return voxels


def generate_all_training_graphs(dataset_output_path: str, voxel_dim: list = (100.0, 100.0, 100.0),
                                 low_degree_threshold: float = 0.01):
    """
    Generator that yields a training graph for each voxel in the dataset.
    :param dataset_output_path: Path to the preprocessed VesselGraph dataset
    :param voxel_dim: The size of the voxel grid
    :param low_degree_threshold: The minimum degree frequency to be considered in the graph
    :return: Yields (voxel_index, node_count, nx_graph, pyg_data) for each valid voxel
    """

    voxels = get_all_voxel_indices(dataset_output_path, voxel_dim)
    print(f'Found {len(voxels)} voxels')

    for voxel_index, node_count in voxels:
        try:
            nx_graph, pyg_data = generate_training_graph(
                dataset_output_path, voxel_dim=voxel_dim,
                low_degree_threshold=low_degree_threshold, voxel_index=voxel_index)
            print(f'Voxel {voxel_index}: {node_count} nodes -> graph with {nx_graph.number_of_nodes()} nodes, '
                  f'{nx_graph.number_of_edges()} edges')
            yield voxel_index, node_count, nx_graph, pyg_data
        except Exception as e:
            print(f'Voxel {voxel_index}: skipped ({e})')
            continue



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
