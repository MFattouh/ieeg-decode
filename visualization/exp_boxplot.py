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
                           # dtype=(float, pd.Categorical))
    for idx, (experiment, csv_path) in enumerate(zip(experiments, csv_pathes)):
        df = pd.read_csv(csv_path)
        results.loc[idx, :] = [df['corr'].mean(), experiment]

    results['corr'] = results['corr'].astype(float)
    exp_type = os.path.basename(os.path.dirname(dataset_dir))
    if exp_type.lower() == 'layers':
        order = sorted(set(experiments), key=lambda exp: int(exp.strip(ascii_letters)))
        x_axes_label = 'Number of Layers'
    elif exp_type.lower() == 'models':
        order = ['SHALLOW', 'DEEP4', 'RNN']
        x_axes_label = 'Architecture'
    else:
        order = None
    g = sns.factorplot(x='exp', y='corr', kind='box', order=order, size=6, data=results)
    g.set_axis_labels(x_axes_label, "corr. coeff.")
    g.savefig(os.path.join(dataset_dir,  exp_type + '.png'))


if __name__ == '__main__':
    exp_plot()
