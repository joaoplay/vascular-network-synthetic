"""
model_learned.py
================
Load the best training checkpoint and grow a full vascular network from a
small seed graph.  Run:

    poetry run python model_learned.py
"""

import glob
import os

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import torch

from settings import OUTPUT_PATH, CHECKPOINTS_DIR_NAME, PROCESSED_DATA_DIR_NAME
from sgg.data import get_signed_distance_between_nodes
from sgg.evaluate import generate_synthetic_graph, get_starting_map
from sgg.graph_data_generator import GraphDataGenerator
from sgg.model import GraphSeq2Seq
from vascular_network.dataset_generation import generate_training_graph

DEVICE = torch.device("cpu")

MODEL_CFG = dict(
    hidden_size=512,
    num_layers=4,
    embedding_size=256,
    is_bidirectional=True,
)

DATA_CFG = dict(
    max_input_paths=4,
    max_paths_for_each_reachable_node=2,
    max_input_path_length=5,
    num_classes=201,
    num_iterations=200,
    remove_duplicates=True,
)

GEN_CFG = dict(
    seed_graph_depth=6,      
    num_iterations=3000,     
    max_loop_distance=1.0,    
)


def find_best_checkpoint(outputs_root: str) -> str:
    """Return the path to the most recently modified checkpoint_best.pt."""
    pattern = os.path.join(outputs_root, "**", CHECKPOINTS_DIR_NAME, "checkpoint_best.pt")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint_best.pt found under '{outputs_root}'. "
            "Run train.py first."
        )
    best = max(candidates, key=os.path.getmtime)
    print(f"Using checkpoint: {best}")
    return best


def load_encoders(full_graph: nx.Graph, preprocessed_data_dir: str):
    """Load the fitted coordinate and radius encoders from the preprocessed cache.

    max_output_nodes was overridden during training to the maximum degree of
    any node in the graph.  We replicate that here so the cache file name
    matches, then confirm by reading the actual data shape.

    Returns (coord_encoder, radius_encoder, max_output_nodes).
    """
    max_output_nodes = max(dict(full_graph.degree()).values())
    generator = GraphDataGenerator(
        graph=full_graph,
        root_dir=preprocessed_data_dir,
        distance_function=get_signed_distance_between_nodes,
        max_output_nodes=max_output_nodes,
        **DATA_CFG,
    )
    _, data_y, coord_encoder, radius_encoder = generator.load()
    max_output_nodes = data_y.shape[2]
    print(f"Encoders loaded. max_output_nodes from data: {max_output_nodes}")
    return coord_encoder, radius_encoder, max_output_nodes


def load_model(checkpoint_path: str, max_output_nodes: int, n_radius_classes: int) -> GraphSeq2Seq:
    model = GraphSeq2Seq(
        n_classes=DATA_CFG["num_classes"],
        max_output_nodes=max_output_nodes,
        n_dimensions=4,
        n_extra_classes=n_radius_classes,
        device=DEVICE,
        **MODEL_CFG,
    ).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.encoder.load_state_dict(checkpoint["encoder"])
    model.decoder.load_state_dict(checkpoint["decoder"])
    model.eval()
    print(
        f"Model loaded  —  iter {checkpoint.get('iter_num', '?')}, "
        f"loss {checkpoint.get('loss', float('nan')):.5f}"
    )
    return model


def _make_cylinder_mesh(p0, p1, radius, n_sides=8):
    axis = p1 - p0
    length = np.linalg.norm(axis)
    if length < 1e-10:
        return None
    axis_norm = axis / length
    perp = np.array([1, 0, 0]) if abs(axis_norm[0]) < 0.9 else np.array([0, 1, 0])
    u = np.cross(axis_norm, perp)
    u /= np.linalg.norm(u)
    v = np.cross(axis_norm, u)
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    circle = radius * (np.outer(np.cos(angles), u) + np.outer(np.sin(angles), v))
    verts = np.vstack([p0 + circle, p1 + circle])
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    ii, jj, kk = [], [], []
    for s in range(n_sides):
        s_next = (s + 1) % n_sides
        ii += [s, s_next]
        jj += [s_next, s_next + n_sides]
        kk += [s + n_sides, s + n_sides]
    return x, y, z, ii, jj, kk


def save_to_html(graph: nx.Graph, original_edges: set, filename: str = "vascular_network_generated.html"):
    """Render the graph as a 3-D cylinder mesh and save to an HTML file.

    Seed edges are drawn in grey; generated edges are coloured by radius class.
    original_edges must be passed explicitly — it is the edge set of the seed
    graph before generation.
    """
    print("Compiling 3D cylinder mesh…")
    nodes = list(graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    coords = np.array([graph.nodes[node]["node_label"][:3] for node in nodes])

    group_defs = [
        {"name": "seed graph", "color": "#cccccc", "r_range": (0, 999),  "is_seed": True},
        {"name": "radius 0–1", "color": "#aec7e8", "r_range": (0, 1),    "is_seed": False},
        {"name": "radius 1–2", "color": "#1f77b4", "r_range": (1, 2),    "is_seed": False},
        {"name": "radius 2–3", "color": "#ff7f0e", "r_range": (2, 3),    "is_seed": False},
        {"name": "radius 3–4", "color": "#2ca02c", "r_range": (3, 4),    "is_seed": False},
        {"name": "radius 4–5", "color": "#d62728", "r_range": (4, 5),    "is_seed": False},
        {"name": "radius > 5", "color": "#8c564b", "r_range": (5, 999),  "is_seed": False},
    ]

    fig = go.Figure()
    for group in group_defs:
        all_x, all_y, all_z, all_i, all_j, all_k = [], [], [], [], [], []
        offset = 0
        found = False

        for u, v, data in graph.edges(data=True):
            r = data.get("avgRadiusAvg") or 0.5
            is_original = (u, v) in original_edges or (v, u) in original_edges
            if group["is_seed"] and not is_original:
                continue
            if not group["is_seed"] and is_original:
                continue
            lo, hi = group["r_range"]
            if not group["is_seed"] and not (lo < r <= hi):
                continue

            found = True
            res = _make_cylinder_mesh(coords[node_to_idx[u]], coords[node_to_idx[v]], max(r, 0.3))
            if res:
                cx, cy, cz, ci, cj, ck = res
                all_x.extend(cx); all_y.extend(cy); all_z.extend(cz)
                all_i.extend(idx + offset for idx in ci)
                all_j.extend(idx + offset for idx in cj)
                all_k.extend(idx + offset for idx in ck)
                offset += len(cx)

        if found:
            opacity = 0.4 if group["is_seed"] else 1.0
            fig.add_trace(go.Mesh3d(
                x=all_x, y=all_y, z=all_z,
                i=all_i, j=all_j, k=all_k,
                color=group["color"], opacity=opacity,
                name=group["name"], showlegend=True,
            ))

    fig.update_layout(
        scene=dict(aspectmode="data", bgcolor="white"),
        title="Vascular Network — grey: seed, colours: AI-generated (by radius)",
    )
    fig.write_html(filename)
    print(f"Saved: {filename}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    preprocessed_data_dir = os.path.join(OUTPUT_PATH, f"{PROCESSED_DATA_DIR_NAME}/")

    full_graph, _ = generate_training_graph(OUTPUT_PATH)

    coord_encoder, radius_encoder, max_output_nodes = load_encoders(full_graph, preprocessed_data_dir)

    checkpoint_path = find_best_checkpoint("outputs/")
    model = load_model(checkpoint_path, max_output_nodes, radius_encoder.n_classes)

    seed_graph, unvisited = get_starting_map(full_graph, depth=GEN_CFG["seed_graph_depth"])
    largest_cc = max(nx.connected_components(seed_graph), key=len)
    seed_graph = seed_graph.subgraph(largest_cc).copy()
    original_edges = set(seed_graph.edges())
    unvisited = [n for n in unvisited if n in seed_graph.nodes()]

    print(f"Seed graph: {len(seed_graph.nodes())} nodes, {len(seed_graph.edges())} edges")
    print(f"Unvisited frontier nodes: {len(unvisited)}")

    generated_graph, _ = generate_synthetic_graph(
        seed_graph=seed_graph,
        graph_seq_2_seq=model,
        categorical_coordinates_encoder=coord_encoder,
        radius_class_encoder=radius_encoder,
        unvisited_nodes=unvisited,
        num_iterations=GEN_CFG["num_iterations"],
        max_input_paths=DATA_CFG["max_input_paths"],
        max_paths_for_each_reachable_node=DATA_CFG["max_paths_for_each_reachable_node"],
        max_input_path_length=DATA_CFG["max_input_path_length"],
        max_output_nodes=max_output_nodes,
        distance_function=get_signed_distance_between_nodes,
        max_loop_distance=GEN_CFG["max_loop_distance"],
        device=DEVICE,
    )

    new_nodes = len(generated_graph.nodes()) - len(seed_graph.nodes())
    print(f"Generated graph: {len(generated_graph.nodes())} nodes, {len(generated_graph.edges())} edges")
    print(f"New nodes added: {new_nodes}")

    save_to_html(generated_graph, original_edges, filename="vascular_network_generated.html")