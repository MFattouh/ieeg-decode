# creates a bar plot to compare the average performance per experiment configuration

import os
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import click
import numpy as np
from string import ascii_letters

TASKS = ['VEL', 'POS']
@click.command()
@click.argument('task_dir', nargs=1, type=click.Path(exists=True))
@click.option('--kind', '-k', type=click.Choice(['violin', 'box', 'bar']), default='box')
@click.option('--palette', '-p', default='muted')
def multi_task_plot(task_dir, kind, palette):
    sns.set_style("darkgrid")
    sns.set_palette("muted")
    sns.set_context("notebook", font_scale=1.5)
    results = create_results_df(task_dir)
    mean_per_model = [(results.loc[results.model == model, 'corr'].mean(), model) for model in set(results.model)]
    order = [model for _, model in sorted(mean_per_model)]
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
    ax.set_ylabel('Corr. Coeff.')
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[0:3], labels[0:3], loc='center left', ncol=1, bbox_to_anchor=(1.0, .5))
    fig.savefig(os.path.join(task_dir, kind + '_multi_task.png'), fancy_box=True, bbox_inches='tight')


def create_results_df(task_dir):
    csv_paths = glob(os.path.join(task_dir, '*/*/*.csv'))
    assert csv_paths, 'No csv files found!'
    models = [os.path.basename(os.path.dirname(path)) for path in csv_paths]
    results = pd.DataFrame(columns=['model'] + TASKS)
    for csv_path, model in zip(csv_paths, models):
        df = pd.read_csv(csv_path)
        df.drop(['mse', 'fold'], axis=1, inplace=True)
        df = df.groupby('day').mean()
        df['model'] = 'MULTI-'+model
        results = results.append(df, ignore_index=True)
    results = results.melt(id_vars=['model'], value_vars=TASKS, var_name='task', value_name='corr')
    results['corr'] = results['corr'].astype(float)
    return results


if __name__ == '__main__':
    multi_task_plot()
