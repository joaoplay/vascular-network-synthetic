import os

import wandb
from sgg.trainer import GraphSeq2SeqTrainer
from utils.visualize import save_graph_html


def evaluate_callback(trainer: GraphSeq2SeqTrainer, every_n_iters: int, output_dir: str = None):
    """
    This callback generates a synthetic graph with the model trained so far and evaluates it against the
    validation data. Optionally saves interactive HTML visualizations.
    :param trainer: A GraphSeq2SeqTrainer model
    :param every_n_iters: How often to evaluate the model
    :param output_dir: If provided, save HTML graph visualizations to this directory
    :return:
    """
    if trainer.iter_num % every_n_iters == 0:
        print(f'Evaluating model at iteration: {trainer.iter_num}...')

        metrics, _ = trainer.evaluate()
        print(f'Validation metrics: {metrics["metrics"]}')
        log_dict = dict(metrics['metrics'])
        plots = metrics['plots']
        for plot_label, plot in plots.items():
            log_dict[plot_label] = plot
        wandb.log(log_dict)

        # Save interactive HTML visualizations
        if output_dir is not None:
            graphs_dir = os.path.join(output_dir, 'graphs')
            os.makedirs(graphs_dir, exist_ok=True)
            for plot_label in ('synthetic_graph', 'seed_graph'):
                fig = plots.get(plot_label)
                if fig is None:
                    continue
                path = os.path.join(graphs_dir, f'{plot_label}_iter_{trainer.iter_num}.html')
                fig.write_html(path)
                print(f'Saved {plot_label} to {path}')


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
    This callback logs the loss and learning rate to wandb
    :param trainer: A GraphSeq2SeqTrainer model
    :param every_n_iters: How often to log the loss to wandb
    :return:
    """
    if trainer.iter_num % every_n_iters == 0:
        current_lr = trainer.encoder_optimizer.param_groups[0]['lr']
        print(f"Loss: {trainer.last_loss_value} | LR: {current_lr:.2e}")
        wandb.log({'Loss': trainer.last_loss_value, 'learning_rate': current_lr})


def save_best_checkpoint_callback(trainer: GraphSeq2SeqTrainer, checkpoint_save_path: str):
    """
    This callback saves the best model checkpoint whenever early stopping detects an improvement,
    and logs early stopping metrics to wandb.
    :param trainer: A GraphSeq2SeqTrainer model
    :param checkpoint_save_path: The path to save the best model checkpoint
    :return:
    """
    if trainer.early_stop_patience is None:
        return

    # Save when patience counter was just reset (new best loss)
    if trainer._patience_counter == 0:
        best_path = os.path.join(checkpoint_save_path, 'checkpoint_best.pt')
        trainer.save_checkpoint(best_path)
        print(f'New best loss: {trainer._best_loss:.6f}. Saved best checkpoint.')
