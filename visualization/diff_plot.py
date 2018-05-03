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
            corr_diff = pd.DataFrame(results[first_exp] - results[second_exp],
                                     columns=['corr_diff']).reset_index()
            corr_diff['exp'] = second_exp
            corr_diff['sub'] = results.index
            df_list.append(corr_diff)

        diff_df = pd.concat(df_list).reset_index()
        diff_df['corr_diff'] = diff_df['corr_diff'].map(lambda x: x * 100)
        g = sns.factorplot(x='exp', y='corr_diff', hue='sub',  legend_out=True, size=6, data=diff_df,
                           kind='strip', jitter=True, order=order)
        ax = sns.pointplot(x='exp', y='corr_diff', ax=g.ax, data=diff_df, join=False, order=order, estimator=np.mean,
                           ci='sd')

        ax.set_ylabel('Corr. Coeff. Difference [%]')
        plt.savefig(os.path.join(dataset_dir,  first_exp + '_diff.png'))


if __name__ == '__main__':
    diff_plot()
