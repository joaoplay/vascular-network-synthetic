import networkx as nx
import numpy as np
import plotly.graph_objects as go
from sgg.radius_classes import R_EDGES

def draw_3d_graph(nx_graph, edges_radius=None, nodes_groups=None, default_radius=3):
    """
    Draw a 3D graph using Plotly. Edges are colored and sized based on their avgRadiusAvg attribute.
    
    :param nx_graph: NetworkX graph to visualize
    :param edges_radius: Optional pre-computed radius list (for backwards compatibility)
    :param nodes_groups: Optional dict of node groups with colors
    :param default_radius: Default radius if not found in edge attributes
    :return: Plotly figure
    """

    nodes_pos = np.array(list(nx.get_node_attributes(nx_graph, "node_label").values())).astype(int)

    coordinates_by_node = {node_id: coordinate for node_id, coordinate in enumerate(nodes_pos)}

    groups = []
    if nodes_groups:
        for group in nodes_groups:
            groups.append({
                'nodes_x': [coordinates_by_node[i][0] for i in group['nodes']],
                'nodes_y': [coordinates_by_node[i][1] for i in group['nodes']],
                'nodes_z': [coordinates_by_node[i][2] for i in group['nodes']],
                'color': group['color'],
                'opacity': group['opacity']
            })

    x_nodes = [coordinates_by_node[i][0] for i in range(len(coordinates_by_node))]  # x-coordinates of nodes
    y_nodes = [coordinates_by_node[i][1] for i in range(len(coordinates_by_node))]  # y-coordinates
    z_nodes = [coordinates_by_node[i][2] for i in range(len(coordinates_by_node))]  # z-coordinates

    # we  need to create lists that contain the starting and ending coordinates of each edge.
    x_edges = []
    y_edges = []
    z_edges = []
    edges_class = []

    edge_list = list(nx_graph.edges())
    if edges_radius is None:
        edge_radius_values = [
            float(nx_graph.edges[edge].get('avgRadiusAvg', default_radius) or default_radius)
            for edge in edge_list
        ]
    else:
        edge_radius_values = [float(radius) for radius in edges_radius]

    # Gather flow values for each edge
    edge_flow_values = [
        float(nx_graph.edges[edge].get('flow', 0) or 0)
        for edge in edge_list
    ]

    # Choose 7 distinguishable colors per class (R_EDGES has 6 boundaries => 7 classes)
    color_map = ["#1f77b4", "#17becf", "#2ca02c", "#bcbd22", "#ff7f0e", "#d62728", "#9467bd"]
    r_edges = np.array(R_EDGES, dtype=float)

    for edge_idx, edge in enumerate(edge_list):
        # format: [beginning,ending,None]
        x_coords = [coordinates_by_node[edge[0]][0], coordinates_by_node[edge[1]][0], None]
        x_edges += x_coords

        y_coords = [coordinates_by_node[edge[0]][1], coordinates_by_node[edge[1]][1], None]
        y_edges += y_coords

        z_coords = [coordinates_by_node[edge[0]][2], coordinates_by_node[edge[1]][2], None]
        z_edges += z_coords

        edge_radi = edge_radius_values[edge_idx]

        # Classify edge by radius
        class_idx = int(np.searchsorted(r_edges, edge_radi, side='left'))
        class_idx = max(0, min(len(color_map) - 1, class_idx))
        edges_class.append(class_idx)



    trace_edges = []
    shown_classes = set()
    labels = ['tiny', 'small', 'medium', 'normal', 'large', 'big', 'huge']

    for edge_idx in range(0, len(edge_list)):
        x_edge = [x_edges[edge_idx * 3], x_edges[edge_idx * 3 + 1], None]
        y_edge = [y_edges[edge_idx * 3], y_edges[edge_idx * 3 + 1], None]
        z_edge = [z_edges[edge_idx * 3], z_edges[edge_idx * 3 + 1], None]

        edge_radius_val = edge_radius_values[edge_idx]
        edge_flow = edge_flow_values[edge_idx]
        class_idx = edges_class[edge_idx]
        edge_color = color_map[class_idx]
        label = labels[class_idx]
        show_legend = class_idx not in shown_classes
        shown_classes.add(class_idx)

        # Create a trace for each edge with class-based coloring
        line_width = min(max(1.0 + 2.5 * np.log1p(edge_radius_val), 1.0), 8.0)
        trace_edges.append(
            go.Scatter3d(x=x_edge, y=y_edge, z=z_edge, mode='lines', 
                         line=dict(color=edge_color, width=line_width),
                         name=f'Class {class_idx}',
                         hovertemplate=f'<b>Vessel</b><br>Radius: {label} <br>Flow: {edge_flow:.2f}<extra></extra>',
                         showlegend=show_legend))

    trace_nodes = []
    if nodes_groups:
        for group in groups:
            trace_nodes.append(go.Scatter3d(x=group['nodes_x'], y=group['nodes_y'], z=group['nodes_z'], mode='markers',
                                            marker=dict(symbol='circle', size=2, color=group['color']),
                                            opacity=group['opacity']
                                            ))
    else:
        # Separate nodes by type: input (red), output (blue), regular (lightgreen)
        node_types = nx.get_node_attributes(nx_graph, 'node_type')
        input_nodes = [n for n, t in node_types.items() if t == 'input']
        output_nodes = [n for n, t in node_types.items() if t == 'output']
        regular_nodes = [n for n in range(len(coordinates_by_node)) if n not in input_nodes and n not in output_nodes]

        if regular_nodes:
            trace_nodes.append(go.Scatter3d(
                x=[coordinates_by_node[n][0] for n in regular_nodes],
                y=[coordinates_by_node[n][1] for n in regular_nodes],
                z=[coordinates_by_node[n][2] for n in regular_nodes],
                mode='markers', marker=dict(symbol='circle', size=2, color='lightgreen'),
                name='Nodes', showlegend=False))

        if input_nodes:
            trace_nodes.append(go.Scatter3d(
                x=[coordinates_by_node[n][0] for n in input_nodes],
                y=[coordinates_by_node[n][1] for n in input_nodes],
                z=[coordinates_by_node[n][2] for n in input_nodes],
                mode='markers', marker=dict(symbol='diamond', size=6, color='red'),
                name='Input', showlegend=True,
                hovertemplate='<b>Input Node</b><extra></extra>'))

        if output_nodes:
            trace_nodes.append(go.Scatter3d(
                x=[coordinates_by_node[n][0] for n in output_nodes],
                y=[coordinates_by_node[n][1] for n in output_nodes],
                z=[coordinates_by_node[n][2] for n in output_nodes],
                mode='markers', marker=dict(symbol='diamond', size=6, color='blue'),
                name='Output', showlegend=True,
                hovertemplate='<b>Output Node</b><extra></extra>'))

    axis = dict(showbackground=False, showline=False, zeroline=False, showgrid=False, showticklabels=True, title='')

    layout = go.Layout(title="Vascular Networks", width=650, height=625, showlegend=False, scene=dict(xaxis=dict(axis),
                                                                                                      yaxis=dict(axis),
                                                                                                      zaxis=dict(axis),
                                                                                                      ),
                       margin=dict(t=100),
                       hovermode='closest')

    data = [*trace_edges, *trace_nodes]
    fig = go.Figure(data=data, layout=layout)

    return fig


def save_graph_html(nx_graph, output_path='graph.html', **kwargs):
    """Save an interactive 3D graph visualization to an HTML file.

    All keyword arguments are forwarded to draw_3d_graph.

    :param nx_graph: NetworkX graph to visualize
    :param output_path: Path for the output HTML file
    """
    fig = draw_3d_graph(nx_graph, **kwargs)
    fig.write_html(output_path)
    print(f'Saved interactive visualization to {output_path}')
