from utils.flow_estimate import annotate_graph_with_flows
from sgg.evaluate import get_starting_map
from vascular_network.dataset_generation import generate_training_graph

#computes the flow, shear stress and pressure drop, for each vessel in the seed graph
graph, i = generate_training_graph('files')
seed_graph, i = get_starting_map(graph, depth=6)
annotate_graph_with_flows(seed_graph)

for u, v in seed_graph.edges():
    e = seed_graph.edges[u, v]
    print(f"({u},{v}): flow={e['flow']:.3f}, shear={e['shear_stress']:.2f}, pressure_drop={seed_graph.nodes[u]['pressure'] - seed_graph.nodes[v]['pressure']:.2f}")