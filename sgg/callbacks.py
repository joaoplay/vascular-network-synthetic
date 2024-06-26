import os

import wandb
from sgg.trainer import GraphSeq2SeqTrainer


def evaluate_callback(trainer: GraphSeq2SeqTrainer, every_n_iters: int):
    """
    This callback generates a synthetic graph with the model trained so far and evaluates it against the
    validation data
    :param trainer: A GraphSeq2SeqTrainer model
    :param every_n_iters: How often to evaluate the model
    :return:
    """
    if trainer.iter_num % every_n_iters == 0:
        print(f'Evaluating model at iteration: {trainer.iter_num}...')

        metrics, _ = trainer.evaluate()
        print(f'Validation metrics: {metrics["metrics"]}')
        # Log validation metrics to wandb
        wandb.log(metrics['metrics'])
        plots = metrics['plots']
        # Send plots to wandb
        for plot_label, plot in plots.items():
            wandb.log({f'{plot_label}': plot})


def save_checkpoint_callback(trainer: GraphSeq2SeqTrainer, every_n_iters: int, checkpoint_save_path: str,
                             save_checkpoint_at_the_end: bool = False):
    """
    This callback saves the model to a checkpoint file
    :param trainer: A GraphSeq2SeqTrainer model
    :param every_n_iters: How often to save the model
    :param checkpoint_save_path: The path to save the model
    :param save_checkpoint_at_the_end: Whether to save the model at the end of training
    :return:
    """
    if save_checkpoint_at_the_end or (every_n_iters is not None and trainer.iter_num % every_n_iters == 0):
        print(f'Saving checkpoint at iteration: {trainer.iter_num}...')
        trainer.save_checkpoint(os.path.join(checkpoint_save_path, f'checkpoint_{trainer.iter_num}.pt'))


def log_loss_callback(trainer: GraphSeq2SeqTrainer, every_n_iters: int):
    """
    This callback logs the loss to wandb
    :param trainer: A GraphSeq2SeqTrainer model
    :param every_n_iters: How often to log the loss to wandb
    :return:
    """
    if trainer.iter_num % every_n_iters == 0:
        print(f"Loss: {trainer.last_loss_value.item()}")
        wandb.log({f'Loss': trainer.last_loss_value.item()})
