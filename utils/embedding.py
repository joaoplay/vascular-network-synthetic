import networkx as nx
import numpy as np
from nodevectors import GGVec


def calculate_embedding_representation(nx_graph: nx.Graph):
    """
    Generate an embedding representation of a graph
    """
    # Learn embedding representation of the generated graph using Node2Vec
    g2v = GGVec(n_components=32)
    g2v.fit(nx_graph)
    # Initializer a numpy array to store the embedding representation of each node
    generated_graph_embedding = np.empty((len(nx_graph.nodes), 32))
    for node in nx_graph.nodes:
        generated_graph_embedding[node] = g2v.predict(node)

    # Return the sum of the embedding representation of each node
    return np.sum(generated_graph_embedding, axis=0)
