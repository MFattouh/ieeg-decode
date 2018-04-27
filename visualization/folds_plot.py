# creates a bar plot of cross-validation folds for different experiments from csv files

import os
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import click


@click.command()
@click.argument('dataset_dir', type=click.Path(exists=True))
def plot_folds(dataset_dir):
    csv_pathes = glob(os.path.join(dataset_dir, '*/*.csv'))
    assert csv_pathes, 'No csv files found!'
    experiments = [os.path.basename(os.path.dirname(path)) for path in csv_pathes]
    df_list = []
    for experiment, csv_path in zip(experiments, csv_pathes):
        df = pd.read_csv(csv_path)
        df.loc[:, 'exp'] = experiment
        df.sort_values(['day', 'exp'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df_list.append(df)
    results = pd.concat(df_list)
    results.reset_index(drop=True, inplace=True)
    days = list(set(results.day.values.tolist()))
    n_folds = len(set(df.fold.values.tolist()))
    exp_type = os.path.basename(os.path.dirname(os.path.dirname(dataset_dir)))
    if exp_type.lower() == 'layers':
        hue_order = ['1Layer', '2Layers', '3Layers']
    else:
        hue_order = None
    for day in days:
        gp = results[results.day == day]
        sns.factorplot(x='fold', y='corr', hue='exp', kind='bar', size=6, data=gp,
                       order=['fold%d' % fold for fold in range(1, n_folds+1)], hue_order=hue_order)
        plt.savefig(os.path.join(dataset_dir, day+'.png'))


if __name__ == '__main__':
    plot_folds()
