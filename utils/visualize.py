import networkx as nx
import numpy as np
import plotly.graph_objects as go
from sgg.radius_classes import r_edges

def smooth_graph_coordinates(nx_graph, iterations=3):
    """
    Aplica Laplacian Smoothing apenas aos nós internos dos vasos (grau 2),
    arredondando esquinas de 90 graus para criar curvas biológicas suaves.
    """
    import numpy as np
    
    # Fazemos uma cópia para não alterar o objeto original em memória inadvertidamente
    smoothed_graph = nx_graph.copy()
    
    for _ in range(iterations):
        new_positions = {}
        for node in smoothed_graph.nodes():
            # Suaviza apenas os nós que são "meio de caminho" (grau 2)
            if smoothed_graph.degree(node) == 2:
                neighbors = list(smoothed_graph.neighbors(node))
                
                # Vai buscar as coordenadas X, Y, Z dos dois vizinhos
                p1 = np.array(smoothed_graph.nodes[neighbors[0]]['node_label'][:3])
                p2 = np.array(smoothed_graph.nodes[neighbors[1]]['node_label'][:3])
                
                # Calcula o ponto médio exato
                new_pos = (p1 + p2) / 2.0
                new_positions[node] = new_pos.tolist()
                
        # Atualiza o grafo com as novas posições suavizadas
        for node, pos in new_positions.items():
            smoothed_graph.nodes[node]['node_label'][:3] = pos
            
    return smoothed_graph


def make_cylinder_mesh(p0, p1, radius, n_sides=8):
    """
    Gera as vértices e as faces de um cilindro 3D entre dois pontos.
    (Lógica extraída do datasetxmin.py)
    """
    v = p1 - p0
    length = np.linalg.norm(v)
    if length == 0:
        return None

    v_dir = v / length
    not_v = np.array([1, 0, 0])
    if np.abs(np.dot(v_dir, not_v)) > 0.99:
        not_v = np.array([0, 1, 0])

    n1 = np.cross(v_dir, not_v)
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(v_dir, n1)

    t = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    circle_pts = np.vstack([np.cos(t), np.sin(t)]).T

    base_circle = p0 + radius * (circle_pts[:, 0:1] * n1 + circle_pts[:, 1:2] * n2)
    top_circle = base_circle + v

    verts = np.vstack([base_circle, top_circle])
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]

    ii, jj, kk = [], [], []
    for s in range(n_sides):
        s_next = (s + 1) % n_sides
        b0, b1 = s, s_next
        t0, t1 = s + n_sides, s_next + n_sides
        # Triângulo 1
        ii.append(b0); jj.append(b1); kk.append(t0)
        # Triângulo 2
        ii.append(b1); jj.append(t1); kk.append(t0)

    return x, y, z, ii, jj, kk


def draw_3d_graph(nx_graph, nodes_groups=None, default_radius=3, edges_radius=None, edges_flow=None):
    """
    Draw a 3D graph using Plotly. Edges are colored by their avgRadiusLabel attribute.

    :param nx_graph: NetworkX graph to visualize
    :param nodes_groups: Optional dict of node groups with colors
    :param default_radius: Default radius if not found in edge attributes
    :return: Plotly figure
    """

    raw_labels = list(nx.get_node_attributes(nx_graph, "node_label").values())
    nodes_pos = np.array([np.array(lbl).flatten()[:3] for lbl in raw_labels]).astype(int)

    """
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


    '''
    # Gather flow values for each edge
    edge_flow_values = [
        float(nx_graph.edges[edge].get('flow', 0) or 0)
        for edge in edge_list
    ]
    '''

    # Choose 7 distinguishable colors per class (r_edges has 6 boundaries => 7 classes)
    color_map = ["#1f77b4", "#17becf", "#2ca02c", "#bcbd22", "#ff7f0e", "#d62728", "#9467bd"]
    r_edges = np.array(r_edges, dtype=float)

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
        #edge_flow = edge_flow_values[edge_idx]
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
                         hovertemplate=f'<b>Vessel</b><br>Radius: {label}',
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
    """
    coordinates_by_node = {node_id: coord for node_id, coord in zip(nx_graph.nodes(), nodes_pos)}

    edge_list = list(nx_graph.edges())
    edge_index_map = {(u, v): i for i, (u, v) in enumerate(edge_list)}

    all_labels = ['tiny', 'small', 'medium', 'normal', 'large', 'big']
    label_colors = {
        'tiny':   "#ff9c07",
        'small':  "#3cff00",
        'medium': "#f00f0f",
        'normal': "#7602fa",
        'large':  "#9e704a",
        'big':    "#ff00ea",
    }
    label_radii = {
        'tiny': 1.0, 'small': 2.5, 'medium': 3.5,
        'normal': 4.5, 'large': 5.5, 'big': 7.0,
    }
    r_edges_arr = np.array(r_edges, dtype=float)

    def _edge_label(edge_data):
        label = edge_data.get('avgRadiusLabel')
        if label is not None:
            return label
        r_val = float(edge_data.get('avgRadiusAvg', default_radius) or default_radius)
        class_idx = int(np.searchsorted(r_edges_arr, r_val, side='left'))
        return all_labels[min(class_idx, len(all_labels) - 1)]

    trace_edges = []

    for label_name, color in label_colors.items():
        all_x, all_y, all_z = [], [], []
        all_i, all_j, all_k = [], [], []
        all_customdata = []
        offset = 0
        has_edges = False

        for u, v, data in nx_graph.edges(data=True):
            if _edge_label(data) != label_name:
                continue

            edge_idx = edge_index_map.get((u, v), edge_index_map.get((v, u)))
            p0 = coordinates_by_node[u]
            p1 = coordinates_by_node[v]
            r = edges_radius[edge_idx] if (edges_radius is not None and edge_idx is not None) else label_radii[label_name]
            flow_label = edges_flow[edge_idx] if (edges_flow is not None and edge_idx is not None) else data.get('flow')

            result = make_cylinder_mesh(p0, p1, radius=r, n_sides=8)
            if result is None:
                continue

            cx, cy, cz, ci, cj, ck = result
            n_verts = len(cx)

            all_x.extend(cx)
            all_y.extend(cy)
            all_z.extend(cz)
            all_i.extend([vert + offset for vert in ci])
            all_j.extend([vert + offset for vert in cj])
            all_k.extend([vert + offset for vert in ck])
            all_customdata.extend([flow_label] * n_verts)

            offset += n_verts
            has_edges = True

        if has_edges:
            mesh = go.Mesh3d(
                x=all_x, y=all_y, z=all_z,
                i=all_i, j=all_j, k=all_k,
                color=color,
                name=label_name,
                showlegend=True,
                customdata=all_customdata,
                hovertemplate=f'<b>Vessel</b><br>Radius: {label_name}<br>Flow: %{{customdata}}<extra></extra>',
            )
            trace_edges.append(mesh)

    axis = dict(showbackground=False, showline=False, zeroline=False, showgrid=False, showticklabels=True, title='')

    layout = go.Layout(title="Vascular Networks", width=800, height=800, showlegend=True, 
                       scene=dict(xaxis=dict(axis),
                                  yaxis=dict(axis),
                                  zaxis=dict(axis),
                                  aspectmode='data'), # Mantém a escala 3D real
                       margin=dict(t=100))

    fig = go.Figure(data=trace_edges, layout=layout)

    return fig


def save_graph_html(nx_graph, output_path='graph.html', smooth_iters=3, **kwargs):    
    """Save an interactive 3D graph visualization to an HTML file.

    All keyword arguments are forwarded to draw_3d_graph.

    :param nx_graph: NetworkX graph to visualize
    :param output_path: Path for the output HTML file
    """
    if smooth_iters > 0:
        nx_graph = smooth_graph_coordinates(nx_graph, iterations=smooth_iters)
    fig = draw_3d_graph(nx_graph, **kwargs)
    fig.write_html(output_path)
    print(f'Saved interactive visualization to {output_path}')
