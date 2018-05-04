# creates a bar plot to compare the average performance per experiment configuration

import os
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import click
import numpy as np
from string import ascii_letters


def exp_plot(x, y, hue, **kwargs):
    ax = plt.gca()
    order = sorted(set(x), key=lambda exp: int(exp.strip(ascii_letters)))
    sns.stripplot(x=x, y=y, hue=hue, jitter=True, order=order, ax=ax)
    sns.pointplot(x=x, y=y, ax=ax, join=False, estimator=np.mean, order=order, ci='sd',
                  markers='D', linestyles=':', errwidth=1, capsize=0.02)


@click.command()
@click.argument('dataset_dir', type=click.Path(exists=True))
def diff_plot(dataset_dir):
    sns.set_style("darkgrid")
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
    results = results.pivot(index='sub', columns='exp', values='corr')
    exp_type = os.path.basename(os.path.dirname(dataset_dir))
    unique_experiments = set(experiments)
    all_exp_list = []
    for first_exp in unique_experiments:
        other_experiments = unique_experiments - set([first_exp])
        df_list = []
        for second_exp in other_experiments:
            if exp_type.lower() == 'layers':
                order = sorted(other_experiments, key=lambda exp: int(exp.strip(ascii_letters)))
            # elif exp_type.lower() == 'models':
            #     order = ['SHALLOW', 'DEEP4', 'RNN']
            else:
                order = None
            corr_diff = pd.DataFrame(results[second_exp] - results[first_exp],
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
    order = sorted(unique_experiments, key=lambda exp: int(exp.strip(ascii_letters)))
    g = sns.FacetGrid(all_exp_df, col='first_exp', sharex=False, size=6, col_order=order
                      )
    g = (g.map(exp_plot, 'exp', 'corr_diff', 'sub')
          .add_legend(title='Subject')
          .set_titles(col_template="{col_name}"))
    g.set_xlabels('')
    # g.set_xticklabels(order, step=1)
    g.set_ylabels('Corr. Coeff. Difference [%]')
    g.savefig(os.path.join(dataset_dir,  'diff.png'))


if __name__ == '__main__':
    diff_plot()
