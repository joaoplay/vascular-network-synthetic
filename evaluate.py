import os

import hydra
import torch
from omegaconf import DictConfig

from settings import OUTPUT_PATH, PROCESSED_DATA_DIR_NAME
from sgg.data import get_signed_distance_between_nodes
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

    # Init a new GraphSeq2Seq model
    model = GraphSeq2Seq(n_classes=cfg.num_classes, max_output_nodes=cfg.paths.max_output_nodes, device=device,
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

    trainer.load_checkpoint(os.path.join(OUTPUT_PATH, 'models', 'default_pretrained.pt'))

    _, steps = trainer.evaluate()


if __name__ == '__main__':
    evaluate_model()
