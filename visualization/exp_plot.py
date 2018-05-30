# creates a bar plot to compare the average performance per experiment configuration

import os
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import click
import numpy as np
from string import ascii_letters

@click.command()
@click.argument('dataset_dir', type=click.Path(exists=True))
@click.option('--kind', '-k', type=click.Choice(['violin', 'box', 'bar']), default='box')
@click.option('-l', '--legend', is_flag=True, help='if passed will add legends outside the plot')
@click.option('--palette', '-p', default='Set2')
def exp_plot(dataset_dir, kind, legend, palette):
    sns.set_context("notebook", font_scale=1.5)
    sns.set_style("darkgrid")
    results = create_results_df(dataset_dir)
    print(results.groupby('exp').median())
    exp_type = os.path.basename(os.path.dirname(dataset_dir))
    unique_experiments = set(results.exp)
    if exp_type.lower() == 'layers':
        order = sorted(unique_experiments, key=lambda exp: int(exp.strip(ascii_letters)))
    else:
        mean_per_exp = [(results.loc[results.exp == experiment, 'corr'].mean(), experiment)
                        for experiment in unique_experiments]
        order = [exp for _, exp in sorted(mean_per_exp)]

    fig, ax = plt.subplots(figsize=(6, 6))
    if kind == 'box':
        sns.boxplot('exp', 'corr', data=results, order=order, ax=ax, palette=palette)
    elif kind == 'violin':
        sns.violinplot('exp', 'corr', data=results, order=order, ax=ax, palette=palette)
    elif kind == 'bar':
        sns.barplot('exp', 'corr', data=results, order=order, ax=ax, palette=palette)
    else:
        raise KeyError

    hue = 'sub' if legend else None
    sns.stripplot('exp', 'corr', ax=ax, data=results, hue=hue, jitter=True, color='0.3', order=order,
                  palette=palette, linewidth=1, edgecolor='gray')
    if legend:
        ax.legend(title='Subject', loc='center', bbox_to_anchor=(1.25, 0.5))
    ax.set_xlabel('')
    ax.set_ylabel('Corr. Coeff.')
    fig.savefig(os.path.join(dataset_dir,  kind + '_' + exp_type + '.png'),  bbox_inches='tight')


def create_results_df(dataset_dir):
    csv_pathes = glob(os.path.join(dataset_dir, '*/*/*.csv'))
    assert csv_pathes, 'No csv files found!'
    experiments = [os.path.basename(os.path.dirname(path)) for path in csv_pathes]
    results = pd.DataFrame(columns=['corr', 'exp', 'sub'])
    for idx, (experiment, csv_path) in enumerate(zip(experiments, csv_pathes)):
        df = pd.read_csv(csv_path)
        subject = os.path.basename(os.path.dirname(os.path.dirname(csv_path)))
        for day in set(df.day.values):
            corr = df.loc[df['day'] == day, 'corr'].mean()
            results = results.append({'corr': corr,
                                      'exp': experiment,
                                      'sub': subject + '_' + day},
                                     ignore_index=True)
    results['corr'] = results['corr'].astype(float)
    return results


if __name__ == '__main__':
    exp_plot()
