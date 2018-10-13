import os
import pandas as pd
from glob import glob
import seaborn as sns
import click
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def exp_plot(x, y, hue, palette=None, **kwargs):
    ax = plt.gca()
    order = ['HYBRID', 'SHALLOW', 'DEEP4', 'RNN']
    if hue is not None:
        color = None
    else:
        color = 'green'
    sns.stripplot(x, y, hue=hue, order=order, dodge=True, ax=ax, jitter=True, size=10,
                  linewidth=1, edgecolor='gray', color=color, palette=palette)

    sns.pointplot(x, y, order=order, ax=ax, join=False, estimator=np.mean, ci='sd', markers='D', linestyles=':',
                  errwidth=3, color='gray', capsize=0.05, scale=2)


@click.command()
@click.argument('task_dirs', nargs=-1, type=click.Path(exists=True))
@click.option('--kind', '-k', type=click.Choice(['violin', 'box', 'bar']), default='box')
@click.option('--palette', '-p', default='muted')
@click.option('-l', '--legend', is_flag=True, help='if passed will add legends outside the plot')
def task_plot(task_dirs, kind, palette, legend):
    sns.set_style("darkgrid")
    sns.set_context("notebook", font_scale=3.2)
    results = create_results_df(task_dirs)
    order = ['ABSVEL', 'XVEL', 'ABSPOS', 'XPOS']
    palette = sns.hls_palette(len(set(results.day.values))) if legend else palette
    hue = 'day' if legend else None
    g = sns.FacetGrid(results, col='task', sharex=False, col_order=order, size=16, aspect=0.6)
    g = (g.map(exp_plot, 'model', 'corr', hue, palette=palette).set_titles(col_template="{col_name}"))
    if legend:
        g.add_legend(title='Recording', fontsize='medium', markerscale=1.5)
    g.set_xlabels('')
    g.set_ylabels('Corr. Coeff.')
    parent = Path(task_dirs[0]).parents[1]
    g.savefig(os.path.join(parent, kind + '_model_per_task.png'), bbox_inches='tight')


def create_results_df(task_dirs):
    results = pd.DataFrame(columns=['corr', 'model', 'task'])
    common_models = set()
    for i, task_dir in enumerate(task_dirs):
        task = Path(task_dir).parents[0].name
        csv_pathes = glob(os.path.join(task_dir, '*/*/*.csv'))
        assert csv_pathes, 'No csv files found!'
        models = [Path(path).parent.name for path in csv_pathes]
        if i == 0:
            common_models = set(models)
        else:
            common_models = common_models & set(models)
        for idx, (model, csv_path) in enumerate(zip(models, csv_pathes)):
            subject = Path(csv_path).parents[1].name
            df = pd.read_csv(csv_path)
            if 'day' not in df.columns.values:
                df.rename(columns={'Unnamed: 0': 'day'}, inplace=True)
            df = df.groupby('day').mean()
            df.reset_index(level=0, inplace=True)
            df['day'] = df.day.apply(lambda rec: '_'.join([rec.split('_')[0], subject, rec.split('_')[1]]))
            df['task'] = task
            df['model'] = model
            results = results.append(df, ignore_index=True)
    results = results[results['model'].isin(list(common_models))]
    results['corr'] = results['corr'].astype(float)
    return results


if __name__ == '__main__':
    task_plot()
