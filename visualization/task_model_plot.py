import os
import pandas as pd
from glob import glob
import seaborn as sns
import click
import matplotlib.pyplot as plt


@click.command()
@click.argument('task_dirs', nargs=-1, type=click.Path(exists=True))
@click.option('--kind', '-k', type=click.Choice(['violin', 'box', 'bar']), default='box')
@click.option('--palette', '-p', default='muted')
def task_plot(task_dirs, kind, palette):
    sns.set_style("darkgrid")
    sns.set_palette("muted")
    sns.set_context("notebook", font_scale=1.5)
    results = create_results_df(task_dirs)
    mean_per_model = [(results.loc[results.model == model, 'corr'].mean(), model) for model in set(results.models)]
    order = [exp for _, exp in sorted(mean_per_model)]
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
    plt.legend(handles[0:2], labels[0:2], loc=2, borderaxespad=0.)
    parent = os.path.dirname(os.path.dirname(task_dirs[0])) if os.path.basename(task_dirs[0]) == '' else \
        os.path.dirname(task_dirs[0])
    fig.savefig(os.path.join(parent, kind + '_model_per_task.png'), bbox_inches='tight')


def create_results_df(task_dirs):
    results = pd.DataFrame(columns=['corr', 'model', 'task'])
    common_models = set()
    for i, task_dir in enumerate(task_dirs):
        task = os.path.basename(task_dir)
        if task == '':
            task = os.path.basename(os.path.dirname(task_dir))
        csv_pathes = glob(os.path.join(task_dir, '*/*/*.csv'))
        assert csv_pathes, 'No csv files found!'
        models = [os.path.basename(os.path.dirname(path)) for path in csv_pathes]
        if i == 0:
            common_models = set(models)
        else:
            common_models = common_models & set(models)
        for idx, (model, csv_path) in enumerate(zip(models, csv_pathes)):
            df = pd.read_csv(csv_path)
            df = df.groupby('day').mean()
            df['task'] = task
            df['model'] = model
            results = results.append(df, ignore_index=True)
    results = results[results['model'].isin(list(common_models))]
    results['corr'] = results['corr'].astype(float)
    return results


if __name__ == '__main__':
    task_plot()
