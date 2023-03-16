import networkx as nx
import numpy as np
from pygments.lexers import go


def draw_3d_graph(nx_graph, edges_radius=None, nodes_groups=None, default_radius=3):
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

    for edge in nx_graph.edges():
        # format: [beginning,ending,None]
        x_coords = [coordinates_by_node[edge[0]][0], coordinates_by_node[edge[1]][0], None]
        x_edges += x_coords

        y_coords = [coordinates_by_node[edge[0]][1], coordinates_by_node[edge[1]][1], None]
        y_edges += y_coords

        z_coords = [coordinates_by_node[edge[0]][2], coordinates_by_node[edge[1]][2], None]
        z_edges += z_coords

    trace_edges = []
    for edge_idx in range(0, len(nx_graph.edges())):
        start_pos = edge_idx * 3
        x_edge = x_edges[start_pos:start_pos + 3]
        y_edge = y_edges[start_pos:start_pos + 3]
        z_edge = z_edges[start_pos:start_pos + 3]

        if edges_radius is not None:
            edge_radius = edges_radius[edge_idx]
        else:
            edge_radius = default_radius

        # Create a trace for the edges
        trace_edges.append(
            go.Scatter3d(x=x_edge, y=y_edge, z=z_edge, mode='lines', line=dict(color='rgba(255, 255, 255, 0.5)',
                                                                               width=edge_radius * 2),
                         hoverinfo='none'))

    trace_nodes = []
    if nodes_groups:
        for group in groups:
            trace_nodes.append(go.Scatter3d(x=group['nodes_x'], y=group['nodes_y'], z=group['nodes_z'], mode='markers',
                                            marker=dict(symbol='circle', size=5, color=group['color']),
                                            opacity=group['opacity']
                                            # line=dict(color='black', width=0.5)),
                                            ))
    else:
        # Create a trace for the nodes
        trace_nodes.append(go.Scatter3d(x=x_nodes, y=y_nodes, z=z_nodes, mode='markers',
                                        marker=dict(symbol='circle', size=5, color='lightgreen'),
                                        # line=dict(color='black', width=0.5)),
                                        ))

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
