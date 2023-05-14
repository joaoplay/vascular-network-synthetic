import vtk
import networkx as nx
import numpy as np

from settings import OUTPUT_PATH
from vascular_network.dataset_generation import generate_training_graph_legacy

# Create a Networkx graph
G, _ = generate_training_graph_legacy(OUTPUT_PATH)

# Assign random coordinates to each node
pos_dict = {node: G.nodes[node]['node_label'] for node in G.nodes}

# Create a vtkPoints object to store the coordinates of the nodes
points = vtk.vtkPoints()
for node in G.nodes:
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
actor.GetProperty().SetOpacity(0.2)

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
def add_node_and_edge():
    # Generate a random position for the new node
    node_list = list(G.nodes)
    idx = np.random.randint(len(node_list))
    existing_node = node_list[idx]
    new_pos = pos_dict[existing_node] + (7 * np.random.randn(3))
    new_node = G.number_of_nodes()
    pos_dict[new_node] = new_pos

    # Add the new node to the graph
    G.add_node(new_node)

    # Connect the new node to an existing node
    G.add_edge(existing_node, new_node)

    # Update the vtkPolyData object with the new node and edge
    points.InsertNextPoint(new_pos)
    line = vtk.vtkLine()
    line.GetPointIds().SetId(0, existing_node)
    line.GetPointIds().SetId(1, new_node)
    lines.InsertNextCell(line)
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    tube_filter.Modified()

    # Refresh the window
    window.Render()

timer_id = interactor.CreateRepeatingTimer(200)
interactor.AddObserver(vtk.vtkCommand.TimerEvent, lambda obj, event: add_node_and_edge())

# Start the interactor
interactor.Initialize()
interactor.Start()