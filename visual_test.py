import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Create a graph
G = nx.Graph()
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])

# Define the positions of the nodes
pos = {0: (0, 0), 1: (1, 0), 2: (1, 1), 3: (0, 1)}

# Create a new figure and axes
fig, ax = plt.subplots(figsize=(5, 5))

# Draw the edges with rounded corners
for u, v in G.edges():
    x1, y1 = pos[u]
    x2, y2 = pos[v]
    dx, dy = x2 - x1, y2 - y1
    dist = ((dx ** 2) + (dy ** 2)) ** 0.5
    if dist == 0:
        continue
    dx /= dist
    dy /= dist
    cx, cy = x1 + dx * dist / 2, y1 + dy * dist / 2
    rad = 0.3 * dist
    angle = np.arctan2(dy, dx)
    arc = plt.Wedge(center=(cx, cy), r=rad, theta1=np.degrees(angle) + 90,
                    theta2=np.degrees(angle) - 90, width=rad, edgecolor='black',
                    facecolor='none')
    ax.add_artist(arc)

# Draw the nodes and labels
nx.draw_networkx_nodes(G, pos, node_size=200, node_color='white', edgecolors='black', ax=ax)
nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif', ax=ax)

# Remove the axes and show the plot
ax.set_axis_off()
plt.show()
