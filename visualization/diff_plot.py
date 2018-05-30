# creates a bar plot to compare the average performance per experiment configuration

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import click
import numpy as np
from string import ascii_letters
from exp_plot import create_results_df


def exp_plot(x, y, hue=None, order=None, **kwargs):
    order = [exp for exp in order if exp in set(x.values)]
    ax = plt.gca()
    color = 'g' if hue is None else None
    sns.stripplot(x=x, y=y, jitter=True, order=order, ax=ax, color=color, hue=hue)
    sns.pointplot(x=x, y=y, ax=ax, join=False, estimator=np.mean, order=order, ci='sd',
                  markers='D', linestyles=':', errwidth=1, capsize=0.02)


@click.command()
@click.argument('dataset_dir', type=click.Path(exists=True))
@click.option('-l', '--legend', is_flag=True, help='if passed will add legends outside the plot')
def diff_plot(dataset_dir, legend):
    sns.set_style("darkgrid")
    sns.set_context("notebook", font_scale=2.2)
    results = create_results_df(dataset_dir)
    unique_experiments = set(results.exp)
    results = results.pivot(index='sub', columns='exp', values='corr')
    exp_type = os.path.basename(os.path.dirname(dataset_dir))
    all_exp_list = []
    for first_exp in unique_experiments:
        other_experiments = unique_experiments - set([first_exp])
        df_list = []
        for second_exp in other_experiments:
            corr_diff = pd.DataFrame(results[first_exp] - results[second_exp],
                                     columns=['corr_diff']).reset_index()
            corr_diff['exp'] = second_exp
            corr_diff['sub'] = results.index
            df_list.append(corr_diff)

        diff_df = pd.concat(df_list).reset_index(drop=True)
        diff_df['first_exp'] = first_exp
        diff_df['corr_diff'] = diff_df['corr_diff'].map(lambda x: x * 100)
        all_exp_list.append(diff_df)

    all_exp_df = pd.concat(all_exp_list).reset_index()
    all_exp_df.drop('index', axis=1, inplace=True)
    if exp_type.lower() == 'layers':
        order = sorted(unique_experiments, key=lambda exp: int(exp.strip(ascii_letters)))
    else:
        mean_per_exp = [(all_exp_df.loc[all_exp_df.exp == experiment, 'corr_diff'].mean(), experiment) for experiment in
                        unique_experiments]
        order = [exp for _, exp in sorted(mean_per_exp)]

    g = sns.FacetGrid(all_exp_df, col='first_exp', sharex=False, size=6, col_order=order)
    if legend:
        g = (g.map(exp_plot, 'exp', 'corr_diff', 'sub', order=order)
              .set_titles(col_template="{col_name}"))
    else:
        g = (g.map(exp_plot, 'exp', 'corr_diff', order=order)
             .set_titles(col_template="{col_name}"))

    if legend:
        g.add_legend(title='Subject')
    g.set_xlabels('')
    g.set_ylabels('Corr. Coeff. Difference [%]')
    g.savefig(os.path.join(dataset_dir,  'diff.png'))


if __name__ == '__main__':
    diff_plot()
