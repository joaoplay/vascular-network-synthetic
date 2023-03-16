import networkx as nx
from torch_geometric.utils import to_networkx


def convert_to_networkx(pyg_graphs):
    """
    Convert a list of PyG graphs to a list of NetworkX graphs
    :param pyg_graphs: List of PyG graphs
    :return:
    """
    networkx_graphs = []
    for graph_idx, graph in enumerate(pyg_graphs):
        node_attrs = {}
        for node_idx, node_attr in enumerate(graph.x):
            # Assign attributes to each node
            node_attrs[node_idx] = {
                'node_label': list(node_attr[0:3]),
            }

        edge_attrs = {}
        for edge_idx in range(len(graph.edge_index[0])):
            start_node = graph.edge_index[0][edge_idx]
            end_node = graph.edge_index[1][edge_idx]

            # Assign attributes to each edge
            edge_attrs[(start_node.item(), end_node.item())] = {
                'radius': graph.edge_attr[edge_idx][2].float().item()}  # Radius

        # Convert to NetworkX graph
        networkx_graph = to_networkx(graph, to_undirected=True)
        nx.set_node_attributes(networkx_graph, node_attrs)
        nx.set_edge_attributes(networkx_graph, edge_attrs)

        networkx_graphs += [networkx_graph]

    return networkx_graphs
