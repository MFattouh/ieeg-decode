# creates a bar plot to compare the average performance per experiment configuration

import os
import matplotlib.pyplot as plt
import seaborn as sns
import click
import numpy as np
from task_model_plot import create_results_df as per_task_df
from multi_task_plot import create_results_df as multi_task_df

TASKS = ['VEL', 'POS']


@click.command()
@click.argument('multi_dir', nargs=1, type=click.Path(exists=True))
@click.argument('task_dirs', nargs=2, type=click.Path(exists=True))
@click.option('--kind', '-k', type=click.Choice(['violin', 'box', 'bar']), default='box')
@click.option('--palette', '-p', default='Spectral')
def multi_vs_per_task(multi_dir, task_dirs, kind, palette):
    multi_df = multi_task_df(multi_dir)
    tasks_df = per_task_df(task_dirs)
    results = multi_df.append(tasks_df, ignore_index=True)
    mean_per_exp = [(results.loc[results.model == model, 'corr'].mean(), model) for model in
                    set(results.model.values)]
    order = [exp for _, exp in sorted(mean_per_exp)]
    sns.set_style("darkgrid")
    sns.set_palette("muted")
    sns.set_context("notebook", font_scale=1.5)
    fig, ax = plt.subplots(figsize=(6, 6))
    if kind == 'box':
        sns.boxplot('task', 'corr', data=results, hue='model', hue_order=order, ax=ax, palette=palette)
    elif kind == 'violin':
        sns.violinplot('task', 'corr', data=results, hue='model', hue_order=order, ax=ax, palette=palette)
    elif kind == 'bar':
        sns.barplot('task', 'corr', data=results, hue='model', hue_order=order, ax=ax, palette=palette)
    else:
        raise KeyError

    sns.stripplot('task', 'corr', hue='model', dodge=True, ax=ax, data=results, jitter=True,
                  palette=palette, linewidth=1, edgecolor='gray', color='0.3', hue_order=order)
    ax.set_xlabel('')
    # ax.set_yticks(list(np.arange(0, 1.0, 0.1)))
    ax.set_ylabel('Corr. Coeff.')
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[0:4], labels[0:4], loc='upper center', bbox_to_anchor=(0.5, 1.18),
               fancybox=True, shadow=True, ncol=2)
    fig.savefig(os.path.join(multi_dir, kind + '_multi_vs_model_per_task.png'), bbox_inches='tight')


if __name__ == '__main__':
    multi_vs_per_task()
