"""
Global boundary condition registry for Secomb flow estimation.

Identifies penetrating vessels (large-radius edges crossing the z=0 plane),
classifies them as arterioles (40 mmHg) or venules (20 mmHg).
Since the whole dataset is treated as one volume, cross-voxel propagation is disabled.
"""

import numpy as np
from vascular_network.dataset import VesselGraphDataset

def _coord_key(x, y, z):
    return (round(float(x), 1), round(float(y), 1), round(float(z), 1))


class VoxelBoundaryRegistry:

    def __init__(self, dataset_output_path, voxel_dim=(1000.0, 1000.0, 1000.0),
                 arteriole_pressure=40.0, venule_pressure=20.0, 
                 radius_threshold=4.0, seed=42):
        self.voxel_dim = voxel_dim
        self.arteriole_pressure = arteriole_pressure
        self.venule_pressure = venule_pressure
        self.radius_threshold = radius_threshold
        self._boundary_pressures = {}

        dataset = VesselGraphDataset(
            root=f'{dataset_output_path}/data', name='synthetic_graph_1',
            use_edge_attr=True, use_atlas=True)
        data = dataset[0].clone()
        coords = data.x[:, :3].numpy()
        edge_index = data.edge_index_undirected
        edge_attr = data.edge_attr_undirected

        min_xyz = coords.min(axis=0)
        max_xyz = coords.max(axis=0)
        print(f"Dataset bounds:")
        print(f"  x: {min_xyz[0]:.2f} to {max_xyz[0]:.2f}")
        print(f"  y: {min_xyz[1]:.2f} to {max_xyz[1]:.2f}")
        print(f"  z: {min_xyz[2]:.2f} to {max_xyz[2]:.2f}")

        z0_nodes = []
        for node_idx in range(len(coords)):
            z = coords[node_idx, 2]
            if z > 1.0:
                continue

            boundary_edges = (edge_index[0] == node_idx) | (edge_index[1] == node_idx)    
            
            node_max_radius = edge_attr[boundary_edges, 2].max().item()
            if node_max_radius >= self.radius_threshold:
                z0_nodes.append(node_idx)

        if z0_nodes:
            rng = np.random.default_rng(seed)
            coin = rng.random(len(z0_nodes)) < 0.5
            for idx, node_idx in enumerate(z0_nodes):
                key = _coord_key(*coords[node_idx])
                pressure = arteriole_pressure if coin[idx] else venule_pressure
                self._boundary_pressures[key] = pressure

        print(f"Fixed z=0 BCs assigned: {len(self._boundary_pressures)}")

    def get_boundary_conditions(self, nx_graph):
        """
        Return Secomb-compatible boundary conditions for the graph.
        """
        bcs = {}
        for node in nx_graph.nodes():
            if nx_graph.degree(node) != 1:
                continue
            label = nx_graph.nodes[node].get('node_label', [0, 0, 0])
            key = _coord_key(label[0], label[1], label[2])
            if key in self._boundary_pressures:
                bcs[node] = {'type': 'pressure', 'value': self._boundary_pressures[key]}
        return bcs
    
    def get_boundary_stats(self, nx_graph):
        bcs = self.get_boundary_conditions(nx_graph)
        total = sum(1 for n in nx_graph.nodes() if nx_graph.degree(n) == 1)
        n_assigned = len(bcs)
        return {
            'total_boundary': total,
            'assigned': n_assigned,
            'penetrating': n_assigned,
            'propagated': 0,
            'unknown': total - n_assigned,
        }