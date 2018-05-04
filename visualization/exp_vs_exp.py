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
def exp_heatmap(dataset_dir):
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
    exp_type = os.path.basename(os.path.dirname(dataset_dir))
    unique_experiments = set(experiments)

    if exp_type.lower() == 'layers':
        order = sorted(unique_experiments, key=lambda exp: int(exp.strip(ascii_letters)))
    elif exp_type.lower() == 'models':
        order = ['SHALLOW', 'DEEP4', 'RNN']
    else:
        order = None

    for first_exp in unique_experiments:
        other_experiments = unique_experiments - set([first_exp])
        for second_exp in other_experiments:
            corr_tabel = results.loc[(results.exp == first_exp) | (results.exp == second_exp)]\
                .pivot(index='sub', columns='exp', values='corr')
            corr_tabel['sub'] = corr_tabel.index

            g = sns.lmplot(x=first_exp, y=second_exp, hue='sub', size=6, data=corr_tabel, fit_reg=False,
                           legend=True, legend_out=True)

            g.ax.plot([0, 1], [0, 1], linestyle=':', color='tab:gray')
            g.set(xticks=np.arange(0, 1.2, 0.2).squeeze().tolist(), xlim=(0, 1))
            g.set(yticks=np.arange(0.2, 1.2, 0.2).squeeze().tolist(), ylim=(0, 1))
            g.savefig(os.path.join(dataset_dir,  first_exp + '_vs_' + second_exp + '.png'))


if __name__ == '__main__':
    exp_heatmap()
