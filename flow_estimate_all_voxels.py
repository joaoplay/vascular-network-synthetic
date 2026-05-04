import os
import csv
import argparse

from utils.flow_estimate import annotate_graph_with_flows
from utils.boundary_conditions import VoxelBoundaryRegistry
from vascular_network.dataset_generation import generate_training_graph, get_all_voxel_indices


def process_voxel(dataset_path: str,
                       output_dir: str,
                       dataset_name: str = 'synthetic_graph_1',
                       voxel_dim: tuple = (1000.0, 1000.0, 1000.0),
                       radius_threshold: float = 4.0,
                       arteriole_pressure: float = 40.0,
                       venule_pressure: float = 20.0,
                       seed: int = 42,
                       voxel_index: int = None):
    os.makedirs(output_dir, exist_ok=True)

    boundary = VoxelBoundaryRegistry(
        dataset_output_path=dataset_path,
        voxel_dim=voxel_dim,
        arteriole_pressure=arteriole_pressure,
        venule_pressure=venule_pressure,
        radius_threshold=radius_threshold,        
        seed=seed
    )

    voxels = get_all_voxel_indices(dataset_path, voxel_dim)
    voxel_index = voxels[0][0]

    graph, pyg_data = generate_training_graph(dataset_path, voxel_dim=voxel_dim, voxel_index=voxel_index)

    print(f"Processing graph {dataset_name} ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)...")
    bcs = boundary.get_boundary_conditions(graph)
        
    annotate_graph_with_flows(graph, boundary_conditions=bcs)


    # Write edges CSV
    edges_csv_path = os.path.join(output_dir, f"{dataset_name}_edges_processed.csv")
    with open(edges_csv_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(["node1id", "node2id","length", "distance", "avgRadiusAvg","roundnessAvg", "curveness", "flow", "pressure_source", "pressure_target"])
        for u, v in graph.edges():
            e = graph.edges[u, v]
            writer.writerow([
                u, v,
                0,
                0,
                f"{e.get('avgRadiusAvg'):.4f}",
                0,
                0,
                f"{e.get('flow'):.6f}",
                f"{graph.nodes[u].get('pressure'):.4f}",
                f"{graph.nodes[v].get('pressure'):.4f}",
            ])

    # Write nodes CSV
    nodes_csv_path = os.path.join(output_dir, f"{dataset_name}_nodes_processed.csv")
    with open(nodes_csv_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(["id", "pos_x", "pos_y", "pos_z",
                         "degree", "isAtSampleBorder"])
        for n in graph.nodes():
            coord = graph.nodes[n]["node_label"]
            degree = graph.degree(n)
            is_border = 1 if degree == 1 else 0
            writer.writerow([
                n,
                f"{coord[0]:.2f}",
                f"{coord[1]:.2f}",
                f"{coord[2]:.2f}",
                degree,
                is_border
            ])

    print(f"Saved processed edges to {edges_csv_path}")
    print(f"Saved processed nodes to {nodes_csv_path}")

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Secomb flow estimation on the entire dataset graph")
    parser.add_argument("--dataset-path", type=str, default="files")
    parser.add_argument("--dataset-name", type=str, default="synthetic_graph_1")
    parser.add_argument("--output-dir", type=str, default="files/flow_estimates")
    parser.add_argument("--voxel-dim", type=float, nargs=3, default=[1000.0, 1000.0, 1000.0])
    parser.add_argument("--radius-threshold", type=float, default=4.0,)
    parser.add_argument("--arteriole-pressure", type=float, default=40.0)
    parser.add_argument("--venule-pressure", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--voxel-index", type=int, default=None)
    args = parser.parse_args()

    process_voxel(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        voxel_dim=tuple(args.voxel_dim),
        radius_threshold=args.radius_threshold,
        arteriole_pressure=args.arteriole_pressure,
        venule_pressure=args.venule_pressure,
        seed=args.seed,
        voxel_index=args.voxel_index)