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
def exp_plot(dataset_dir):
    csv_pathes = glob(os.path.join(dataset_dir, '*/*/*.csv'))
    assert csv_pathes, 'No csv files found!'
    experiments = [os.path.basename(os.path.dirname(path)) for path in csv_pathes]
    results = pd.DataFrame(index=np.arange(len(csv_pathes)), columns=['corr', 'exp'])
    for idx, (experiment, csv_path) in enumerate(zip(experiments, csv_pathes)):
        df = pd.read_csv(csv_path)
        results.loc[idx, :] = [df['corr'].mean(), experiment]

    results['corr'] = results['corr'].astype(float)
    exp_type = os.path.basename(os.path.dirname(dataset_dir))
    if exp_type.lower() == 'layers':
        order = sorted(set(experiments), key=lambda exp: int(exp.strip(ascii_letters)))
    else:
        mean_per_exp = [(results.loc[results.exp == experiment, 'corr'].mean(), experiment) for experiment in set(experiments)]
        order = [exp for _, exp in sorted(mean_per_exp)]
    g = sns.factorplot(x='exp', y='corr', kind='box', order=order, size=6, data=results)
    g.set_axis_labels('', "Corr. Coeff.")
    g.savefig(os.path.join(dataset_dir,  exp_type + '.png'))


if __name__ == '__main__':
    exp_plot()
