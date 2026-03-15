import csv
from utils.flow_estimate import annotate_graph_with_flows
from vascular_network.dataset_generation import generate_training_graph

#BALBc_no1 dataset
graph, i = generate_training_graph('files')
print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

annotate_graph_with_flows(graph)

with open("flow_estimate_dataset.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["node_id", "  x", "  y", "  z", "  neighbor_id", "  flow", "  pressure drop"])
    for u, v in graph.edges():
        coord_u = graph.nodes[u]["node_label"]
        pu = graph.nodes[u]["pressure"]
        flow = graph.edges[u, v]["flow"]
        writer.writerow([
            f"{u:>7}",
            f"{coord_u[0]:>8.2f}",
            f"{coord_u[1]:>8.2f}",
            f"{coord_u[2]:>8.2f}",
            f"{v:>13}",
            f"{flow:>12.6f}",
            f"{pu:>12.4f}",
        ])

print("Saved to flow_estimate_dataset.csv")

