# creates a bar plot to compare the average performance per experiment configuration

import os
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import click
import numpy as np

@click.command()
@click.argument('dataset_dir', type=click.Path(exists=True))
def plot_folds(dataset_dir):
    csv_pathes = glob(os.path.join(dataset_dir, '*/*/*.csv'))
    assert csv_pathes, 'No csv files found!'
    experiments = [os.path.basename(os.path.dirname(path)) for path in csv_pathes]
    results = pd.DataFrame(index=np.arange(len(csv_pathes)), columns=['corr', 'exp'])
    for idx, (experiment, csv_path) in enumerate(zip(experiments, csv_pathes)):
        df = pd.read_csv(csv_path)
        results.loc[idx, :] = [df['corr'].mean(), experiment]

    exp_type = os.path.basename(os.path.dirname(dataset_dir))
    if exp_type.lower() == 'layers':
        order = ['1Layer', '2Layers', '3Layers']
    else:
        order = None

    sns.factorplot(x='exp', y='corr', kind='bar', order=order, size=6, data=results)
    plt.savefig(os.path.join(dataset_dir,  exp_type + '.png'))


if __name__ == '__main__':
    plot_folds()
