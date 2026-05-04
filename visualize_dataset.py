"""
Interactive dataset visualizer for identifying inlet and outlet nodes.

Usage:
    poetry run python visualize_dataset.py [--voxel]

By default loads the FULL BALBc_no1 dataset (~3.5M nodes, ~5.3M edges).
Use --voxel to load only the 100x100x100 voxel crop (1385 nodes) for quick testing.

Exports an interactive HTML file you can open in a browser to rotate/pan/zoom.
Boundary nodes (degree=1) are analyzed by face and radius to find penetrating vessels.
Red = large-radius boundary (arteriole candidates), Blue = medium, Magenta = small.
"""

import json
import sys
import numpy as np
import pyvista as pv
import torch
import torch_geometric
from torch_geometric.data import Data

from settings import OUTPUT_PATH
from vascular_network.dataset import VesselGraphDataset

# ===== CONFIGURATION =====
USE_VOXEL = "--voxel" in sys.argv
TUBE_RADIUS_SCALE = 1.0       # Scale avgRadiusAvg for tube thickness
MIN_TUBE_RADIUS = 0.5         # Minimum tube radius
BOUNDARY_SPHERE_SCALE = 3.0   # Multiplier for boundary sphere size relative to local radius

pv.OFF_SCREEN = True
pv.global_theme.background = [0.05, 0.05, 0.1]

print("Loading dataset...")
dataset = VesselGraphDataset(root=f'{OUTPUT_PATH}/data', name='BALBc_no1', use_edge_attr=True, use_atlas=False)
data = dataset[0].clone()

if USE_VOXEL:
    print("Using 100x100x100 voxel crop...")
    from vascular_network.dataset_generation import generate_training_graph
    from utils.util import set_seed
    set_seed(60)
    nx_graph, pyg_data = generate_training_graph(OUTPUT_PATH)
    # Convert back to tensor form
    node_coords = np.array([nx_graph.nodes[n]['node_label'] for n in sorted(nx_graph.nodes())])
    nodes_list = sorted(nx_graph.nodes())
    node_id_map = {n: i for i, n in enumerate(nodes_list)}
    edges_src, edges_dst, edge_radii = [], [], []
    for u, v, d in nx_graph.edges(data=True):
        edges_src.append(node_id_map[u])
        edges_dst.append(node_id_map[v])
        edge_radii.append(d.get('avgRadiusAvg', 1.0))
    edges_src = np.array(edges_src)
    edges_dst = np.array(edges_dst)
    edge_radii = np.array(edge_radii)
else:
    print("Loading FULL dataset (this may take a minute)...")
    data_und = Data(x=data.x, edge_index=data.edge_index_undirected, edge_attr=data.edge_attr_undirected)

    # Get largest connected component
    lcc = torch_geometric.transforms.LargestConnectedComponents()
    ri = torch_geometric.transforms.RemoveIsolatedNodes()
    ri(data_und)
    data_und = lcc(data_und)

    node_coords = data_und.x[:, 0:3].numpy()
    ei = data_und.edge_index.numpy()

    # Deduplicate undirected edges (keep u < v)
    mask = ei[0] < ei[1]
    edges_src = ei[0, mask]
    edges_dst = ei[1, mask]
    edge_radii = data_und.edge_attr[mask, 2].float().numpy()  # radius is attribute 2

n_nodes = len(node_coords)
n_edges = len(edges_src)
print(f"Graph: {n_nodes} nodes, {n_edges} edges")

# ===== Compute node degree =====
degree = np.zeros(n_nodes, dtype=int)
np.add.at(degree, edges_src, 1)
np.add.at(degree, edges_dst, 1)

# ===== Bounding box analysis =====
bbox_min = node_coords.min(axis=0)
bbox_max = node_coords.max(axis=0)
bbox_range = bbox_max - bbox_min
print(f"\nBounding box:")
print(f"  X: {bbox_min[0]:.1f} - {bbox_max[0]:.1f}  (range {bbox_range[0]:.1f})")
print(f"  Y: {bbox_min[1]:.1f} - {bbox_max[1]:.1f}  (range {bbox_range[1]:.1f})")
print(f"  Z: {bbox_min[2]:.1f} - {bbox_max[2]:.1f}  (range {bbox_range[2]:.1f})")

# ===== Boundary nodes (degree=1) =====
boundary_mask = degree == 1
boundary_indices = np.where(boundary_mask)[0]

# Get max radius of edge connected to each boundary node
boundary_edge_radius = np.zeros(n_nodes)
for i in range(n_edges):
    s, d = edges_src[i], edges_dst[i]
    r = edge_radii[i]
    if boundary_mask[s]:
        boundary_edge_radius[s] = max(boundary_edge_radius[s], r)
    if boundary_mask[d]:
        boundary_edge_radius[d] = max(boundary_edge_radius[d], r)

# Classify boundary nodes by face
face_names = ['-X', '+X', '-Y', '+Y', '-Z', '+Z']
face_vals = [
    (0, bbox_min[0]), (0, bbox_max[0]),
    (1, bbox_min[1]), (1, bbox_max[1]),
    (2, bbox_min[2]), (2, bbox_max[2]),
]

boundary_face = {}
for idx in boundary_indices:
    c = node_coords[idx]
    dists = [abs(c[axis] - val) for axis, val in face_vals]
    boundary_face[idx] = face_names[np.argmin(dists)]

# Stats per face
print(f"\n{'='*80}")
print(f"BOUNDARY NODE ANALYSIS - {len(boundary_indices)} nodes total")
print(f"{'='*80}")
print(f"{'Face':<6} {'Count':>6} {'Avg Radius':>12} {'Max Radius':>12}")
print(f"{'-'*40}")
face_groups = {name: [] for name in face_names}
for idx in boundary_indices:
    face_groups[boundary_face[idx]].append(idx)

for fn in face_names:
    group = face_groups[fn]
    if not group:
        print(f"{fn:<6} {0:>6}")
        continue
    radii = boundary_edge_radius[group]
    print(f"{fn:<6} {len(group):>6} {np.mean(radii):>12.2f} {np.max(radii):>12.2f}")

# Classify by radius percentiles
all_br = boundary_edge_radius[boundary_indices]
q75 = np.percentile(all_br, 75)
q50 = np.percentile(all_br, 50)
# If Q75 == Q50 (many ties), shift Q75 up
if q75 <= q50:
    q75 = q50 + 0.5

print(f"\nRadius thresholds: Q50={q50:.2f}, Q75={q75:.2f}")

arteriole_idx = boundary_indices[all_br >= q75]
venule_idx = boundary_indices[(all_br >= q50) & (all_br < q75)]
other_idx = boundary_indices[all_br < q50]

print(f"Arteriole candidates (red, r>={q75:.1f}): {len(arteriole_idx)}")
print(f"Venule candidates (blue, r>={q50:.1f}): {len(venule_idx)}")
print(f"Other boundary (magenta, r<{q50:.1f}): {len(other_idx)}")

# Print top penetrating candidates
sorted_boundary = sorted(boundary_indices, key=lambda i: boundary_edge_radius[i], reverse=True)
print(f"\n{'Rank':>4} {'Node':>8} {'Radius':>8} {'Face':>6}  {'X':>9} {'Y':>9} {'Z':>9}")
print(f"{'-'*60}")
for rank, idx in enumerate(sorted_boundary[:30]):
    c = node_coords[idx]
    print(f"{rank+1:>4} {idx:>8} {boundary_edge_radius[idx]:>8.2f} {boundary_face[idx]:>6}  "
          f"{c[0]:>9.1f} {c[1]:>9.1f} {c[2]:>9.1f}")

# ===== Build pyvista mesh =====
print("\nBuilding 3D mesh...")

# Build lines for edges: pyvista format is [n_pts, id0, id1, n_pts, id0, id1, ...]
cells = np.column_stack([
    np.full(n_edges, 2, dtype=int),
    edges_src,
    edges_dst,
]).ravel()

cell_types = np.full(n_edges, pv.CellType.LINE)
grid = pv.UnstructuredGrid(cells, cell_types, node_coords.astype(float))
grid.cell_data['radius'] = edge_radii

# For tube rendering, we need polydata with lines
lines_array = np.column_stack([
    np.full(n_edges, 2, dtype=int),
    edges_src,
    edges_dst,
]).ravel()

poly = pv.PolyData(node_coords.astype(float))
poly.lines = lines_array

# Point-based radius for tube filter (max of connected edges)
point_radius = np.full(n_nodes, MIN_TUBE_RADIUS)
for i in range(n_edges):
    r = max(MIN_TUBE_RADIUS, edge_radii[i] * TUBE_RADIUS_SCALE)
    s, d = edges_src[i], edges_dst[i]
    point_radius[s] = max(point_radius[s], r)
    point_radius[d] = max(point_radius[d], r)

poly.point_data['TubeRadius'] = point_radius

# Tube filter
if n_edges < 50000:
    print("Applying tube filter (small enough for full tubes)...")
    tubes = poly.tube(scalars='TubeRadius', absolute=True, n_sides=8, capping=True)
else:
    print(f"Dataset has {n_edges} edges — using line rendering for performance.")
    print("(Tube filter would be too slow. Line width indicates radius.)")
    tubes = None

# Build the plotter
plotter = pv.Plotter(off_screen=True, window_size=[1600, 1000])

if tubes is not None:
    plotter.add_mesh(tubes, color='gainsboro', specular=0.3, specular_power=20, opacity=1.0)
else:
    # Color lines by radius  
    plotter.add_mesh(poly, scalars=point_radius, cmap='bone', line_width=2,
                     render_lines_as_tubes=True, scalar_bar_args={'title': 'Radius (μm)'})

# Add boundary spheres
def add_spheres(plotter, indices, color, radius_mult):
    if len(indices) == 0:
        return
    centers = node_coords[indices]
    radii = np.maximum(boundary_edge_radius[indices] * radius_mult, 3.0)
    cloud = pv.PolyData(centers)
    glyphs = cloud.glyph(geom=pv.Sphere(radius=1.0), scale=False, orient=False)
    # Scale by adjusting sphere sizes - use a uniform size for visibility
    avg_r = np.mean(radii)
    sphere = pv.Sphere(radius=avg_r, center=[0, 0, 0])
    glyphs = cloud.glyph(geom=sphere, scale=False, orient=False)
    plotter.add_mesh(glyphs, color=color, opacity=0.9)

add_spheres(plotter, arteriole_idx, 'red', BOUNDARY_SPHERE_SCALE)
add_spheres(plotter, venule_idx, 'blue', BOUNDARY_SPHERE_SCALE)
add_spheres(plotter, other_idx, 'magenta', BOUNDARY_SPHERE_SCALE * 0.5)

plotter.add_text(
    f"{'VOXEL' if USE_VOXEL else 'FULL'} Vascular Network: {n_nodes} nodes, {n_edges} edges\n"
    f"Red={len(arteriole_idx)} arteriole cand.  Blue={len(venule_idx)} venule cand.  "
    f"Magenta={len(other_idx)} other",
    position='upper_left', font_size=10, color='white'
)

# Export interactive HTML
output_html = "vascular_network_interactive.html"
print(f"Exporting interactive HTML to {output_html}...")
plotter.export_html(output_html)
print(f"Saved {output_html} — open in browser to rotate/pan/zoom!")

# Also save a static PNG
plotter.show(screenshot="vascular_network_view1.png")
print("Saved vascular_network_view1.png")

# Save boundary info
boundary_info = {
    "dataset": "BALBc_no1",
    "mode": "voxel_100x100x100" if USE_VOXEL else "full",
    "n_nodes": int(n_nodes),
    "n_edges": int(n_edges),
    "boundary_nodes": [int(i) for i in boundary_indices],
    "penetrating_arteriole_candidates": [
        {"node_id": int(i), "radius": float(boundary_edge_radius[i]),
         "face": boundary_face[i], "coords": node_coords[i].tolist()}
        for i in arteriole_idx
    ],
    "penetrating_venule_candidates": [
        {"node_id": int(i), "radius": float(boundary_edge_radius[i]),
         "face": boundary_face[i], "coords": node_coords[i].tolist()}
        for i in venule_idx
    ],
    "other_boundary": [
        {"node_id": int(i), "radius": float(boundary_edge_radius[i]),
         "face": boundary_face[i], "coords": node_coords[i].tolist()}
        for i in other_idx
    ],
    "inlets": [],
    "outlets": [],
}

with open("boundary_conditions.json", "w") as f:
    json.dump(boundary_info, f, indent=2)
print("Saved boundary_conditions.json")
print("\nDone!")
