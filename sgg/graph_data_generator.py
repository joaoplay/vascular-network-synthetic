import os

import networkx as nx
import numpy as np
import torch
from matplotlib import pyplot as plt

from sgg.data import generate_training_samples_for_node
from utils.scale_factor_categorical_encoder import ScaleFactorCategoricalCoordinatesEncoder
from utils.torch import unique


class GraphDataGenerator:
    """
    Generate training data for a graph. This is a random walk based approach. For each node in the graph, a set of random
    paths are extracted. The remaining neighboring nodes are considered as the prediction targets. Each point corresponds
    to the relative position between a given node and the next one in the path. The relative position are encoded to
    categorical variables by means of a CategoricalCoordinatesEncoder. Depending on the number of iterations to
    generate the data, this process can become very time-consuming. The data is then saved to the disk and can be
    automatically loaded though the load() method (the name of the files are automatically generated using a unique
    combination of the parameters of this class).
    """

    def __init__(self, graph: nx.Graph, max_input_paths: int, max_paths_for_each_reachable_node: int,
                 max_input_path_length: int, distance_function, max_output_nodes: int,
                 num_classes: int, num_iterations: int, remove_duplicates: bool, root_dir: str) -> None:
        """
        :param graph: The graph from which the data will be generated
        :param max_input_paths: The maximum number of paths to be extract for each node in a single iteration
        :param max_paths_for_each_reachable_node: The maximum number of paths to be extracted for each reachable node
        :param max_input_path_length: The maximum length of each path
        :param distance_function: The distance function to be used to compute the distance between two nodes
        :param max_output_nodes: The maximum number of output nodes to be considered for each node.
        :param num_classes: Total number of classes to be used for the categorical coordinates' encoder.
        :param num_iterations: Number of iteration to generate data for each node.
        :param remove_duplicates: Whether to remove duplicate paths or not. Probably not a good idea to set this to False.
        In such case, there would exist duplicated samples in the dataset.
        :param root_dir:
        """
        super().__init__()
        self.graph: nx.Graph = graph
        self.max_input_paths = max_input_paths
        self.max_paths_for_each_reachable_node = max_paths_for_each_reachable_node
        self.max_input_path_length = max_input_path_length
        self.distance_function = distance_function
        self.max_output_nodes = max_output_nodes
        self.num_classes = num_classes
        self.num_iterations = num_iterations
        self.remove_duplicates = remove_duplicates
        self.root_dir = root_dir

    def load(self):
        """
        Load the training data from disk. Load the categorical coordinates encoder as well. The data returned by this
        method is the same as the one generated by the "generate()" method.
        :return:
        """
        print(f'Loading from existing preprocessed data in {self.data_x_path} and {self.data_y_path}.')

        # Load categorical encoder. The file is assumed to be in the same directory as the data and the name is
        # always the same.
        categorical_coordinates_encoder = ScaleFactorCategoricalCoordinatesEncoder(self.num_classes)
        # Load training data
        data_x = torch.load(self.data_x_path)
        data_y = torch.load(self.data_y_path)

        print(f'Dataset has shape {data_x.shape} and {data_y.shape}')

        return data_x, data_y, categorical_coordinates_encoder

    def generate(self):
        """
        Generate training data for the graph, performing num_iterations. Each iteration consists in
        generating a single pair of input and output nodes for every node in the graph. Note that, theoretically, large
        graph may require more iterations to better capture the structural patterns of the graph.
        :return:
        """

        print(f"Generating {self.num_iterations} iterations of training data for graph with "
              f"{len(self.graph.nodes)} nodes")

        # 1 - Generate training data
        input_data = []
        prediction_data = []
        for _ in range(self.num_iterations):
            # Initialize input and output data for the current iteration
            cur_input_data = []
            cur_prediction_data = []

            # For each node in the graph, we will generate a set of paths and output nodes
            for node in self.graph.nodes:
                # Generate training samples for the current node
                x, y = generate_training_samples_for_node(self.graph, node, self.max_input_paths,
                                                          self.max_paths_for_each_reachable_node,
                                                          self.max_input_path_length, self.max_output_nodes,
                                                          self.distance_function)

                cur_input_data += x
                cur_prediction_data += y

            # Append the current iteration data to the global data
            input_data.append(cur_input_data)
            prediction_data.append(cur_prediction_data)

        # Concatenate the data from all iterations
        input_data = np.concatenate(input_data, axis=0)
        prediction_data = np.concatenate(prediction_data, axis=0)

        # Convert input_data and prediction_data to tensors
        input_data = torch.tensor(input_data)
        prediction_data = torch.tensor(prediction_data)

        # 2 - Convert relative positions to classes

        # Create and fit categorical encoder
        categorical_coordinates_encoder = ScaleFactorCategoricalCoordinatesEncoder(n_categories=self.num_classes)
        # Encode input_data and prediction_data
        input_data = categorical_coordinates_encoder.transform(input_data)
        prediction_data = categorical_coordinates_encoder.transform(prediction_data)

        # Draw histogram of the input data and prediction data
        plt.hist(input_data.flatten(), bins=100)
        plt.title('Input data histogram')
        plt.show()

        plt.hist(prediction_data.flatten(), bins=100)
        plt.title('Prediction data histogram')
        plt.show()

        print(f'Input data min: {input_data.min()}')
        print(f'Input data max: {input_data.max()}')
        print(f'Prediction data min: {prediction_data.min()}')
        print(f'Prediction data max: {prediction_data.max()}')

        print(f'Dataset has shape data_x:{input_data.shape} and data_y:{prediction_data.shape}')

        # Remove duplicates. This is necessary because a random process is applied to generate the training data. We
        # have no guarantee that the same pair of input paths and output nodes will not be generated more than once.
        if self.remove_duplicates:
            print(f'Removing duplicates...')
            input_data, unique_indices = unique(input_data, dim=0)
            prediction_data = prediction_data[unique_indices]

            print(f'Dataset after removing duplicates has shape data_x:{input_data.shape} '
                  f'and data_y:{prediction_data.shape}')

        # Save data to disk
        torch.save(input_data, self.data_x_path)
        torch.save(prediction_data, self.data_y_path)

        return input_data, prediction_data, categorical_coordinates_encoder

    @property
    def file_name_prefix(self):
        """
        Build the file name prefix for the data files. It allows to uniquely identify the data generated by the
        current configuration.
        :return:
        """
        return f'{self.max_input_paths}_{self.max_paths_for_each_reachable_node}_' \
               f'{self.max_input_path_length}_{self.max_output_nodes}_{self.num_classes}' \
               f'_{self.num_iterations}_{self.remove_duplicates}'

    @property
    def data_x_path(self):
        """
        Build the path to the file containing the input data.
        :return:
        """
        return os.path.join(self.root_dir, f'{self.file_name_prefix}_data_x.pt')

    @property
    def data_y_path(self):
        """
        Build the path to the file containing the output data.
        :return:
        """
        return os.path.join(self.root_dir, f'{self.file_name_prefix}_data_y.pt')

    @property
    def categorical_encoder_path(self):
        """
        Build the path to the file containing the categorical encoder.
        :return:
        """
        return os.path.join(self.root_dir, f'{self.file_name_prefix}_categorical_coordinates_encoder.pt')
