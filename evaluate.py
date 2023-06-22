import os
import sys

import hydra
import numpy as np
import torch
import vtk
from omegaconf import DictConfig

from settings import OUTPUT_PATH, PROCESSED_DATA_DIR_NAME
from sgg.data import get_signed_distance_between_nodes
from sgg.evaluate import get_starting_map, reset_subgraph_indexes, random_subgraph
from sgg.graph_data_generator import GraphDataGenerator
from sgg.model import GraphSeq2Seq
from sgg.trainer import GraphSeq2SeqTrainer
from vascular_network.dataset_generation import generate_training_graph


@hydra.main(config_path="configs", config_name="default_config", version_base="1.2")
def evaluate_model(cfg: DictConfig):
    # Set device from the environment variable
    device = torch.device(cfg.trainer.device)

    # Generate a training graph from VesselGraph
    training_graph, _ = generate_training_graph(OUTPUT_PATH)

    # Override max output nodes to be the maximum between the config and the maximum degree across all
    # nodes in the graph
    cfg.paths.max_output_nodes = max([training_graph.degree(node) for node in training_graph.nodes()])

    # Init a new GraphSeq2Seq model
    model = GraphSeq2Seq(n_classes=cfg.num_classes + 1, max_output_nodes=cfg.paths.max_output_nodes, device=device,
                         **cfg.model)

    preprocessed_data_dir = os.path.join(OUTPUT_PATH, f'{PROCESSED_DATA_DIR_NAME}/')

    # Create a GraphDataGenerator responsible for generating the sequential training data from a graph.
    graph_data_generator = GraphDataGenerator(graph=training_graph, root_dir=preprocessed_data_dir,
                                              distance_function=get_signed_distance_between_nodes,
                                              num_classes=cfg.num_classes,
                                              num_iterations=cfg.num_preprocessing_iterations,
                                              remove_duplicates=cfg.remove_duplicates, **cfg.paths)

    data_x, data_y, cat_coordinates_encoder = graph_data_generator.load()

    # Init a trainer for the GraphSeq2Seq model. We don't specify a train dataset nor class weights because we are
    # using the trainer only for evaluation purposes.
    # Init a trainer for the GraphSeq2Seq model
    trainer = GraphSeq2SeqTrainer(model=model, train_dataset=None, graph=training_graph,
                                  distance_function=get_signed_distance_between_nodes,
                                  categorical_coordinates_encoder=cat_coordinates_encoder,
                                  class_weights=None, ignore_index=cfg.num_classes, **cfg.evaluator,
                                  **cfg.paths,
                                  **cfg.trainer)

    trainer.load_checkpoint(os.path.join(OUTPUT_PATH, 'models', 'checkpoint_4000.pt'))

    _, steps = trainer.evaluate()

    print(steps)

    # interactive_evaluation(steps, cfg.evaluator.seed_graph_depth)


def interactive_evaluation(steps, max_depth):
    # Create a Networkx graph
    raw, _ = generate_training_graph(OUTPUT_PATH)

    # Get a seed from the graph and considering a maximum starting_seed_depth
    # G = random_subgraph(raw, max_depth=12)
    G, _ = get_starting_map(raw, depth=max_depth)
    #G, _ = random_subgraph(raw, max_depth=max_depth)

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
    actor.GetProperty().SetColor(0.0, 1.0, 0.0)
    actor.GetProperty().SetOpacity(1.0)

    # Create a vtkRenderer object to display the actor
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(0.0, 0.0, 0.0)

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

    timer_id = interactor.CreateRepeatingTimer(50)
    interactor.AddObserver(vtk.vtkCommand.TimerEvent, lambda obj, event: add_node_and_edge(steps.pop(0) if steps else None))

    # Start the interactor
    interactor.Initialize()
    interactor.Start()

if __name__ == '__main__':
    evaluate_model()
