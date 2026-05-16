from typing import Callable, Dict, Optional

import networkx as nx
import torch
from networkx import Graph
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader

from sgg.evaluate import get_starting_map, generate_synthetic_graph, compute_graph_comparison_metrics
from sgg.model import GraphSeq2Seq
from utils.categorical_coordinates_encoder import CategoricalCoordinatesEncoder
from utils.radius_class_encoder import RadiusClassEncoder
from utils.flow_class_encoder import FlowClassEncoder
from utils.visualize import draw_3d_graph

# Macros for available callbacks
ON_BATCH_END = 'on_batch_end'
ON_TRAIN_END = 'on_train_end'
ON_EVAL_END = 'on_eval_end'


class GraphSeq2SeqTrainer:
    def __init__(self, model: GraphSeq2Seq, train_dataset: Optional[Dataset], graph: Graph, max_input_paths: int,
                 max_paths_for_each_reachable_node: int, max_input_path_length: int, max_output_nodes: int,
                 distance_function: Callable, max_loop_distance: int, synthetic_graph_gen_iterations: int,
                 seed_graph_depth: int, categorical_coordinates_encoder: CategoricalCoordinatesEncoder,
                 radius_class_encoder: Optional[RadiusClassEncoder],
                 ignore_index: Optional[int],
                 lr: float, max_iters: int, batch_size: int, device: str,
                 xyz_class_weights: Optional[torch.Tensor] = None,
                 radius_class_weights: Optional[torch.Tensor] = None,
                 flow_class_weights: Optional[torch.Tensor] = None,
                 radius_ignore_index: Optional[int] = None,
                 flow_ignore_index: Optional[int] = None,
                 radius_loss_weight: float = 1.0,
                 flow_loss_weight: float = 1.0,
                 flow_class_encoder: Optional[FlowClassEncoder] = None,
                 early_stop_patience: Optional[int] = None,
                 **kwargs):
        """
        This class offers a training loop for a Graph Sequence-to-Sequence model. It is responsible for training
        the model, while providing callbacks to perform custom actions during the training process. The metrics
        are stored as attributes of the class and can be access inside a callback.
        :param model: A GraphSeq2Seq model
        :param train_dataset: A PyTorch Dataset containing the training data
        :param graph: A NetworkX graph containing the original graph
        :param max_input_paths: The maximum number of input paths to be extracted from each node
        :param max_paths_for_each_reachable_node: The maximum number of paths to be extracted from each reachable node
        :param max_input_path_length: The maximum length of each input path
        :param max_output_nodes: The maximum number of output nodes
        :param distance_function: The distance function to be used to compute the distance between two nodes
        :param max_loop_distance: The maximum distance between two nodes to be considered a loop. This is used
                                  during the generation of synthetic graphs.
        :param synthetic_graph_gen_iterations: How many iterations to perform when generating synthetic graphs
        :param categorical_coordinates_encoder: A CategoricalCoordinatesEncoder instance. It must be already trained
                                                with the dataset.
        :param lr: The learning rate
        :param max_iters: The maximum training iterations
        :param batch_size: The batch size
        :param device: The device to be used for training
        :param radius_ignore_index: The index to be ignored in the radius loss calculation
        :param flow_ignore_index: The index to be ignored in the flow loss calculation
        """
        self.model: GraphSeq2Seq = model
        self.train_dataset = train_dataset
        self.graph = graph
        self.max_input_paths = max_input_paths
        self.max_paths_for_each_reachable_node = max_paths_for_each_reachable_node
        self.max_input_path_length = max_input_path_length
        self.max_output_nodes = max_output_nodes
        self.distance_function = distance_function
        self.max_loop_distance = max_loop_distance
        self.batch_size = batch_size
        self.synthetic_graph_gen_iterations = synthetic_graph_gen_iterations
        self.seed_graph_depth = seed_graph_depth
        self.categorical_coordinates_encoder = categorical_coordinates_encoder
        self.radius_class_encoder = radius_class_encoder
        self.flow_class_encoder = flow_class_encoder
        self.encoder_optimizer = optim.Adam(self.model.encoder.parameters(), lr=lr)
        self.decoder_optimizer = optim.Adam(self.model.decoder.parameters(), lr=lr)
        self.early_stop_patience = early_stop_patience
        self._best_loss = float('inf')
        self._patience_counter = 0
        self._should_stop = False
        # Use separate CrossEntropy losses for spatial (xyz) and radius channels.
        # This allows assigning a dedicated weight to vessel thickness learning.
        xyz_weights = xyz_class_weights.to(device=device) if xyz_class_weights is not None else None
        radius_weights = radius_class_weights.to(device=device) if radius_class_weights is not None else None
        flow_weights = flow_class_weights.to(device=device) if flow_class_weights is not None else None

        print("xyz_class_weights", xyz_weights)
        print("radius_class_weights", radius_weights)
        print("flow_class_weights", flow_weights)
        self.xyz_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=xyz_weights)
        self.radius_loss = nn.CrossEntropyLoss(ignore_index=radius_ignore_index, weight=radius_weights)
        self.flow_loss = nn.CrossEntropyLoss(ignore_index=flow_ignore_index, weight=flow_weights) if flow_ignore_index is not None else None
        self.radius_loss_weight = radius_loss_weight
        self.flow_loss_weight = flow_loss_weight
        print(f"radius_loss_weight: {self.radius_loss_weight}")
        print(f"flow_loss_weight: {self.flow_loss_weight}")
        self.max_iters = max_iters
        self.device = device
        self.callbacks: Dict = {}
        # Attributes to be used for logging and evaluation
        self.iter_num = 0
        self.last_loss_value = 0
        self.xyz_loss_value = 0
        self.radius_loss_value = 0
        self.flow_loss_value = 0

    def train(self):
        """
        The training loop. It iterates over the training dataset and performs a forward pass and a backward pass
        for each batch. It also triggers the ON_BATCH_END callbacks.
        :return:
        """
        self.model.train()
        # Create a simple data loader
        dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        data_iter = iter(dataloader)
        # A new training process is starting. Reset the iteration number and the last loss value.
        self.last_loss_value = 0
        self.iter_num = 0
        while True:
            try:
                # Get the next batch
                batch = next(data_iter)
            except StopIteration:
                # No more data left. Create a new iterator and get the next batch.
                data_iter = iter(dataloader)
                batch = next(data_iter)

            # Move the batch x and y to the device
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            #select the last path for teacher forcing. Padding is prepended at the start, so the
            # last path index is always a real (non-padded) path. All real paths within the same
            # training sample predict the same output nodes, so there is no benefit from randomness.
            # y shape: (batch_size, n_paths, max_output_nodes, n_dimensions)
            y_selected_paths = y[:, -1, :]

            # Forward pass through the encoder. This is done in batches.
            decoder_output = self.model(x, y_selected_paths)

            # Calculate separate losses for spatial dimensions and radius.
            n_dimensions = y_selected_paths.size(-1)
            n_classes = decoder_output.size(-1)

            decoder_output = decoder_output.view(-1, n_dimensions, n_classes)
            y_target = y_selected_paths.reshape(-1, n_dimensions)

            #for xyz
            spatial_dims = 3
            xyz_output = decoder_output[:, :spatial_dims, :].reshape(-1, n_classes)
            xyz_target = y_target[:, :spatial_dims].reshape(-1)
            xyz_loss = self.xyz_loss(xyz_output, xyz_target)

            #for radius
            n_radius_classes = self.model.n_radius_classes
            radius_output = decoder_output[:, 3, :n_radius_classes]
            radius_target = y_target[:, 3]
            radius_loss = self.radius_loss(radius_output, radius_target)

            flow_loss = None
            self.last_loss_value = xyz_loss + (self.radius_loss_weight * radius_loss)
            if n_dimensions > 4 and self.flow_loss is not None:
                n_flow_classes = self.model.n_flow_classes
                flow_output = decoder_output[:, 4, :n_flow_classes]
                flow_target = y_target[:, 4]
                flow_loss = self.flow_loss(flow_output, flow_target)
                self.last_loss_value = self.last_loss_value + (self.flow_loss_weight * flow_loss)

            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            self.last_loss_value.backward()
            torch.nn.utils.clip_grad_norm_(self.model.encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.decoder.parameters(), 1.0)
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

            # Convert to scalar to free computation graph
            self.last_loss_value = self.last_loss_value.item()
            self.xyz_loss_value = xyz_loss.item()
            self.radius_loss_value = radius_loss.item()
            self.flow_loss_value = flow_loss.item() if flow_loss is not None else 0.0

            # Early stopping check
            if self.early_stop_patience is not None:
                if self.last_loss_value < self._best_loss:
                    self._best_loss = self.last_loss_value
                    self._patience_counter = 0
                else:
                    self._patience_counter += 1
                    if self._patience_counter >= self.early_stop_patience:
                        print(f'Early stopping at iteration {self.iter_num}. '
                              f'Best loss: {self._best_loss:.6f}, '
                              f'Current loss: {self.last_loss_value:.6f}')
                        self._should_stop = True

            # Trigger ON_BATCH_END callbacks
            self.trigger_callbacks(ON_BATCH_END)

            if self._should_stop:
                break

            if self.max_iters is not None and self.iter_num > self.max_iters:
                # The maximum number of iterations has been reached. Stop the training.
                break

            # Increase iteration number
            self.iter_num += 1

        self.trigger_callbacks(ON_TRAIN_END)

    @torch.no_grad()
    def evaluate(self):
        """
        Evaluate the model on the seed_graph generated from the original graph. The evaluation is performed
        by generating a synthetic graph from the seed_graph and comparing it with the original graph.
        :return:
        """
        # Get a seed from the graph and considering a maximum starting_seed_depth
        seed_graph, unvisited_nodes = get_starting_map(self.graph, depth=self.seed_graph_depth)

        # Generate a synthetic graph from the previously obtained seed.
        synth_graph, steps = generate_synthetic_graph(seed_graph=seed_graph, unvisited_nodes=unvisited_nodes,
                                                      graph_seq_2_seq=self.model,
                                                      categorical_coordinates_encoder=self.categorical_coordinates_encoder,
                                                      radius_class_encoder=self.radius_class_encoder,
                                                      flow_class_encoder=self.flow_class_encoder,
                                                      max_input_paths=self.max_input_paths,
                                                      max_paths_for_each_reachable_node=self.max_paths_for_each_reachable_node,
                                                      max_input_path_length=self.max_input_path_length,
                                                      max_output_nodes=self.max_output_nodes,
                                                      distance_function=self.distance_function,
                                                      num_iterations=self.synthetic_graph_gen_iterations,
                                                      max_loop_distance=self.max_loop_distance, device=self.device)


        synth_graph = nx.convert_node_labels_to_integers(synth_graph, label_attribute='old_label')
        fig1 = draw_3d_graph(synth_graph)

        seed_graph = nx.convert_node_labels_to_integers(seed_graph, label_attribute='old_label')
        fig2 = draw_3d_graph(seed_graph)


        # Calculate metrics
        metrics = compute_graph_comparison_metrics(synth_graph, self.graph)
        metrics['plots']['synthetic_graph'] = fig1
        metrics['plots']['seed_graph'] = fig2
        


        return metrics, steps

    def add_callback(self, on_event: str, callback: Callable, *args, **kwargs):
        """
        Add a callback to be triggered when a specific event occurs.
        :param on_event: An event as defined in this file
        :param callback: The callback function
        :param args:
        :param kwargs:
        :return:
        """

        # This is wrapped to allow the callback to receive additional arguments when adding to the trainer.
        def wrapped_callback(*cb_args, **cb_kwargs):
            return callback(*cb_args, *args, **cb_kwargs, **kwargs)

        self.callbacks.setdefault(on_event, [])
        self.callbacks[on_event].append(wrapped_callback)

    def trigger_callbacks(self, on_event: str):
        """
        Trigger all the callbacks that are registered for a specific event.
        :param on_event:
        :return:
        """
        callbacks = self.callbacks.get(on_event, [])
        for callback in callbacks:
            callback(self)

    def save_checkpoint(self, checkpoint_save_path: str):
        """
        Save a checkpoint of the model, the optimizer and other relevant information.
        :param checkpoint_save_path: Path to save the checkpoint
        :return:
        """
        torch.save({
            'encoder': self.model.encoder.state_dict(),
            'decoder': self.model.decoder.state_dict(),
            'encoder_optimizer': self.encoder_optimizer.state_dict(),
            'decoder_optimizer': self.decoder_optimizer.state_dict(),
            'iter_num': self.iter_num,
            'loss': self.last_loss_value
        }, checkpoint_save_path)

    def load_checkpoint(self, checkpoint_path):
        """
        Load a checkpoint from a file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.encoder.load_state_dict(checkpoint['encoder'])
        self.model.decoder.load_state_dict(checkpoint['decoder'])
        self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
        self.decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
        self.iter_num = checkpoint['iter_num']
        self.last_loss_value = checkpoint['loss']

