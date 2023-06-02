import networkx as nx
import numpy as np
import vtk

from settings import OUTPUT_PATH
from sgg.evaluate import get_starting_map, random_subgraph, reset_subgraph_indexes
from utils.util import set_seed
from vascular_network.dataset_generation import generate_training_graph

# Set random seed
set_seed(42)

steps = [(8, 447, [918.0, 3943.0, 703.3120403289795]), (8, 448, [921.275390625, 3943.0, 703.3120403289795]),
         (8, 448, None), (8, 448, None), (8, 448, None), (8, 448, None), (8, 448, None), (8, 448, None), (8, 448, None),
         (8, 448, None), (41, 449, [969.0, 3965.9633502960205, 705.0]), (41, 449, None), (41, 449, None),
         (233, 450, [880.421989440918, 3918.0, 729.0]), (233, 450, None), (233, 450, None), (233, 450, None),
         (401, 451, [949.7290403289795, 3942.58, 750.333]), (401, 452, [954.738460723877, 3942.58, 754.957080657959]),
         (401, 453, [944.9122907562256, 3942.58, 754.957080657959]), (401, 453, None), (401, 453, None),
         (401, 453, None), (401, 453, None), (401, 453, None), (401, 453, None),
         (401, 454, [945.490299407959, 3939.3046103286742, 754.957080657959]), (433, 432, None),
         (433, 455, [932.942390625, 4005.724760131836, 760.926700592041]),
         (433, 456, [932.942390625, 4000.33, 760.926700592041]), (433, 456, None), (433, 456, None), (433, 456, None),
         (433, 456, None), (433, 456, None), (433, 456, None), (433, 456, None),
         (438, 457, [914.0638799667358, 3952.073299407959, 768.3581104278564]),
         (438, 458, [919.651309967041, 3952.073299407959, 768.3581104278564]), (438, 458, None),
         (438, 459, [919.651309967041, 3952.073299407959, 761.6146602630615]), (438, 459, None),
         (438, 460, [924.6607303619385, 3952.073299407959, 767.5874309539795]), (438, 460, None), (438, 460, None),
         (438, 460, None), (438, 460, None), (446, 461, [910.3759202957153, 3994.0, 767.0]),
         (447, 462, [913.3759202957153, 3943.0, 703.697380065918]), (447, 462, None),
         (447, 463, [925.321460723877, 3943.0, 705.046070098877]), (447, 463, None), (447, 462, None), (447, 462, None),
         (447, 462, None), (447, 63, None), (447, 63, None), (447, 63, None),
         (454, 464, [945.490299407959, 3939.3046103286742, 758.0398004608154]),
         (458, 465, [919.651309967041, 3950.146598815918, 768.3581104278564]),
         (458, 466, [916.5685901641846, 3950.146598815918, 769.321460723877]),
         (458, 467, [922.926700592041, 3950.146598815918, 769.321460723877]), (458, 467, None), (458, 467, None),
         (458, 467, None), (458, 467, None), (458, 467, None), (458, 467, None), (458, 466, None),
         (459, 468, [917.724609375, 3952.073299407959, 761.6146602630615]), (459, 468, None), (459, 468, None),
         (459, 468, None), (459, 468, None), (459, 468, None), (459, 468, None), (459, 468, None), (459, 468, None),
         (459, 468, None), (460, 469, [924.6607303619385, 3952.073299407959, 762.578010559082]), (460, 469, None),
         (460, 469, None), (460, 469, None), (460, 469, None),
         (461, 470, [910.3759202957153, 3994.0, 769.3120403289795]), (462, 63, None),
         (465, 471, [919.651309967041, 3950.146598815918, 771.4408302307129]),
         (468, 472, [915.797908782959, 3952.073299407959, 761.6146602630615]), (468, 472, None), (468, 466, None),
         (468, 466, None), (468, 466, None), (468, 466, None), (468, 472, None), (468, 472, None),
         (468, 473, [912.7151889801025, 3950.146598815918, 760.073299407959]), (468, 473, None),
         (471, 474, [919.651309967041, 3950.146598815918, 773.1748600006104]), (471, 474, None), (471, 474, None),
         (471, 474, None), (474, 475, [919.651309967041, 3950.146598815918, 776.2575798034668]), (474, 475, None),
         (474, 475, None), (474, 475, None), (474, 475, None), (474, 475, None), (474, 475, None), (474, 475, None),
         (474, 475, None), (474, 475, None), (475, 476, [918.302619934082, 3950.146598815918, 777.9916095733643]),
         (475, 476, None), (475, 477, [915.0272302627563, 3950.724609375, 777.9916095733643]),
         (475, 478, [915.0272302627563, 3939.935089111328, 777.9916095733643]), (475, 478, None), (475, 478, None),
         (475, 478, None), (475, 479, [918.8806304931641, 3937.6230487823486, 777.9916095733643]), (475, 479, None),
         (475, 480, [910.9811601638794, 3937.6230487823486, 777.9916095733643]), (476, 477, None), (476, 477, None),
         (476, 477, None), (476, 477, None), (476, 477, None),
         (476, 481, [911.9445095062256, 3950.146598815918, 779.7256393432617]), (476, 481, None), (476, 481, None),
         (476, 481, None), (476, 481, None), (479, 482, [918.8806304931641, 3943.5958194732666, 768.9361190795898]),
         (479, 483, [918.8806304931641, 3931.650279045105, 768.9361190795898]),
         (479, 484, [911.9445104598999, 3931.650279045105, 768.9361190795898]), (479, 484, None), (479, 482, None),
         (479, 485, [917.7246112823486, 3943.5958194732666, 779.9183101654053]),
         (479, 486, [922.5413608551025, 3943.5958194732666, 779.7256393432617]), (479, 486, None), (479, 486, None),
         (479, 487, [922.5413608551025, 3943.5958194732666, 784.7350597381592])]

# Create a Networkx graph
raw, _ = generate_training_graph(OUTPUT_PATH)

# Get a seed from the graph and considering a maximum starting_seed_depth
# G = random_subgraph(raw, max_depth=12)
G, _ = get_starting_map(raw, depth=12)

G = reset_subgraph_indexes(G)

# Convert node_label attribute to numpy array
for node in G.nodes:
    G.nodes[node]['node_label'] = np.array(G.nodes[node]['node_label'])

# Assign random coordinates to each node
pos_dict = {node: G.nodes[node]['node_label'] for node in G.nodes}

# Convert coordinates to numpy array
pos = np.array([pos_dict[node] for node in G.nodes])
# Print min and max for each coordinate
print('x: ', np.min(pos[:, 0]), np.max(pos[:, 0]))
print('y: ', np.min(pos[:, 1]), np.max(pos[:, 1]))
print('z: ', np.min(pos[:, 2]), np.max(pos[:, 2]))

# Create a vtkPoints object to store the coordinates of the nodes
points = vtk.vtkPoints()
for node in G.nodes:
    print(node, pos_dict[node])
    points.InsertNextPoint(pos_dict[node])

# Create a vtkCellArray object to store the edges of the graph
lines = vtk.vtkCellArray()
for edge in G.edges:
    line = vtk.vtkLine()
    line.GetPointIds().SetId(0, edge[0])
    line.GetPointIds().SetId(1, edge[1])
    lines.InsertNextCell(line)

# Create a vtkPolyData object from the graph
polydata = vtk.vtkPolyData()
polydata.SetPoints(points)
polydata.SetLines(lines)

# Create a vtkTubeFilter to add tubes around each edge
tube_filter = vtk.vtkTubeFilter()
tube_filter.SetInputData(polydata)
tube_filter.SetRadius(0.5)
tube_filter.SetNumberOfSides(10)

# Create a vtkPolyDataMapper object to map the tubes to a 3D surface
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(tube_filter.GetOutputPort())

# Create a vtkActor object to define the properties of the 3D surface
actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetColor(1.0, 0.0, 0.0)
actor.GetProperty().SetOpacity(1.0)

# Create a vtkRenderer object to display the actor
renderer = vtk.vtkRenderer()
renderer.AddActor(actor)
renderer.SetBackground(1.0, 1.0, 1.0)

# Create a vtkRenderWindow object to display the renderer
window = vtk.vtkRenderWindow()
window.AddRenderer(renderer)

# Create a vtkRenderWindowInteractor object to handle user input
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(window)


# Define the function to add a new node and edge to the graph
def add_node_and_edge(action):
    if action[2]:
        new_pos = action[2]
        # Add the new node to the graph
        G.add_node(action[1])
        points.InsertNextPoint(new_pos)

    # Generate a random position for the new node
    node_list = list(G.nodes)
    existing_node = node_list[action[0]]
    pos_dict[action[1]] = action[2]

    # Connect the new node to an existing node
    G.add_edge(action[0], action[1])

    # Update the vtkPolyData object with the new node and edge
    line = vtk.vtkLine()
    line.GetPointIds().SetId(0, existing_node)
    line.GetPointIds().SetId(1, action[1])
    lines.InsertNextCell(line)
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    tube_filter.Modified()

    # Refresh the window
    window.Render()


timer_id = interactor.CreateRepeatingTimer(1000)
interactor.AddObserver(vtk.vtkCommand.TimerEvent, lambda obj, event: add_node_and_edge(steps.pop(0)))

# Start the interactor
interactor.Initialize()
interactor.Start()
