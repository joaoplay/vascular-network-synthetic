import os

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import TensorDataset

from settings import OUTPUT_PATH, CHECKPOINTS_DIR_NAME, PROCESSED_DATA_DIR_NAME, WANDB_PROJECT_NAME, \
    MODELS_DIR_NAME
from sgg.callbacks import evaluate_callback, save_checkpoint_callback, log_loss_callback
from sgg.data import get_signed_distance_between_nodes
from sgg.evaluate import degree_analysis
from sgg.graph_data_generator import GraphDataGenerator
from sgg.model import GraphSeq2Seq
from sgg.trainer import GraphSeq2SeqTrainer, ON_BATCH_END, ON_TRAIN_END
from utils.torch import compute_class_weights
from utils.util import set_seed, create_directory
from vascular_network.dataset_generation import generate_training_graph


@hydra.main(config_path="configs", config_name="default_config", version_base="1.2")
def train_model(cfg: DictConfig):
    """
    This method is responsible for training a Graph Sequence-to-Sequence model. After reading the Hydra configuration
    files, a dataset is generated from subset of the VesselGraph. A sampling process is applied to the dataset to
    extract sequential steps of the graph evolution. For a given node, a set of random paths with a maximum length are
    extracted and the remaining nodes are considered as the prediction targets.
    :param cfg:
    :return:
    """
    # This is a hack. Somehow hydra is not setting the working directory to the output directory.
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    hydra_cwd = hydra_cfg['runtime']['output_dir']

    # Don't forget to set WANDB_API_KEY environment variable. This is required for wandb to work.
    wandb.init(project=WANDB_PROJECT_NAME, name=cfg.run_name, tags=cfg.tags, config=OmegaConf.to_container(cfg,
                                                                                                           resolve=True))

    # Set the run output path
    models_output_dir = os.path.join(hydra_cwd, MODELS_DIR_NAME)
    checkpoints_dir = os.path.join(hydra_cwd, CHECKPOINTS_DIR_NAME)
    preprocessed_data_dir = os.path.join(OUTPUT_PATH, f'{PROCESSED_DATA_DIR_NAME}/')

    # Create directories
    create_directory(OUTPUT_PATH)
    create_directory(models_output_dir)
    create_directory(checkpoints_dir)
    create_directory(preprocessed_data_dir)

    # Set random seed
    set_seed(cfg.seed, cfg.trainer.device)
    # Set device from the environment variable
    device = torch.device(cfg.trainer.device)

    # Generate a training graph from VesselGraph
    training_graph, _ = generate_training_graph(OUTPUT_PATH)

    # Override max output nodes to be the maximum between the config and the maximum degree across all
    # nodes in the graph
    cfg.paths.max_output_nodes = max([training_graph.degree(node) for node in training_graph.nodes()])

    # Calculate average node degree
    avg_node_degree = sum([training_graph.degree(node) for node in training_graph.nodes()]) / len(training_graph.nodes())

    # Create a GraphDataGenerator responsible for generating the sequential training data from a graph.
    graph_data_generator = GraphDataGenerator(graph=training_graph, root_dir=preprocessed_data_dir,
                                              distance_function=get_signed_distance_between_nodes,
                                              num_classes=cfg.num_classes,
                                              num_iterations=cfg.num_preprocessing_iterations,
                                              remove_duplicates=True, **cfg.paths)

    # Load or generate data. Apart from the data, the corresponding categorical coordinates encoder is returned.
    # It will be necessary to generate new predictions and evaluate the model.
    if cfg.force_rebuild_data:
        data_x, data_y, cat_coordinates_encoder = graph_data_generator.generate()
    else:
        try:
            data_x, data_y, cat_coordinates_encoder = graph_data_generator.load()
        except FileNotFoundError:
            print('Data not found. Generating new data.')
            data_x, data_y, cat_coordinates_encoder = graph_data_generator.generate()

    # The weight of each class is used to balance the loss function. Most of the neighboring nodes are close to
    # each other, resulting in a high number of transitions close to 0.
    class_weights = compute_class_weights(data_y, cfg.num_classes + 1)
    # Compute node degree statistics for the original training graph. This is important to later compare with
    # the synthetic graphs. The corresponding graphs are logged to wandb.
    training_graph_degree_analysis = degree_analysis(nx_graph=training_graph)
    wandb.log({f'Ground Truth Degree Analysis': wandb.Image(training_graph_degree_analysis[0])})

    # Create a dataloader for the training data
    dataset = TensorDataset(data_x, data_y)

    # Init a new GraphSeq2Seq model
    model = GraphSeq2Seq(n_classes=cfg.num_classes + 1, max_output_nodes=cfg.paths.max_output_nodes, device=device,
                         **cfg.model)

    # Init a trainer for the GraphSeq2Seq model
    trainer = GraphSeq2SeqTrainer(model=model, train_dataset=dataset, graph=training_graph,
                                  distance_function=get_signed_distance_between_nodes,
                                  categorical_coordinates_encoder=cat_coordinates_encoder,
                                  class_weights=class_weights, ignore_index=cfg.num_classes, **cfg.evaluator,
                                  **cfg.paths, **cfg.trainer)

    # Add a set of callbacks to: print the training loss; generate a synthetic graph and perform a comparison
    # with the validation data; save the model to a checkpoint file.
    trainer.add_callback(ON_BATCH_END, log_loss_callback, every_n_iters=cfg.log_loss_every_n_iters)
    trainer.add_callback(ON_BATCH_END, evaluate_callback, every_n_iters=cfg.eval_every_n_iters)
    trainer.add_callback(ON_TRAIN_END, save_checkpoint_callback, every_n_iters=cfg.save_checkpoint_every_n_iters,
                         checkpoint_save_path=checkpoints_dir, save_checkpoint_at_the_end=cfg.save_at_the_end)

    print(f'Training model: {cfg.run_name} on device: {device}...')
    # Train the model
    trainer.train()

    wandb.finish()


if __name__ == '__main__':
    train_model()
