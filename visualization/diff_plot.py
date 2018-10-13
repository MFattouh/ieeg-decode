# creates a bar plot to compare the average performance per modeleriment configuration

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import click
import numpy as np
from exp_plot import create_results_df


def model_plot(x, y, hue, order=None, palette=None, **kwargs):
    order = [model for model in order if model in set(x.values)]
    ax = plt.gca()
    color = 'green' if palette is None else None
    sns.stripplot(x=x, y=y, jitter=True, order=order, ax=ax, color=color, hue=hue, size=8, palette=palette)
    if palette is None:
        color = None
    else:
        color = 'gray'
        palette = None

    sns.pointplot(x=x, y=y, ax=ax, join=False, estimator=np.mean, order=order, ci='sd', palette=palette,
                  markers='D', linestyles=':', errwidth=3, color=color, capsize=0.02, scale=2)


@click.command()
@click.argument('dataset_dir', type=click.Path(exists=True))
@click.option('-l', '--legend', is_flag=True, help='if passed will add legends outside the plot')
def diff_plot(dataset_dir, legend):
    sns.set_style("darkgrid")
    sns.set_context("notebook", font_scale=3)
    results = create_results_df(dataset_dir)
    unique_models = set(results.model)
    results = results.pivot(index='sub', columns='model', values='corr')
    model_type = os.path.basename(os.path.dirname(dataset_dir))
    all_model_list = []
    for first_model in unique_models:
        other_models = unique_models - set([first_model])
        df_list = []
        for second_model in other_models:
            corr_diff = pd.DataFrame(results[first_model] - results[second_model],
                                     columns=['corr_diff']).reset_index()
            corr_diff['model'] = second_model
            corr_diff['sub'] = results.index
            df_list.append(corr_diff)

        diff_df = pd.concat(df_list).reset_index(drop=True)
        diff_df['first_model'] = first_model
        diff_df['corr_diff'] = diff_df['corr_diff'].map(lambda x: x * 100)
        all_model_list.append(diff_df)

    all_model_df = pd.concat(all_model_list).reset_index()
    all_model_df.drop('index', axis=1, inplace=True)
    if model_type.lower() == 'layers':
        order = sorted(unique_models, key=lambda model: int(model.strip(ascii_letters)))
    else:
        order = ['HYBRID', 'SHALLOW', 'DEEP4', 'RNN']

    hue = 'sub' if legend else None
    palette = sns.hls_palette(len(set(all_model_df['sub']))) if legend else None
    g = sns.FacetGrid(all_model_df, col='first_model', col_order=order, sharex=False, size=12, aspect=0.6)
    g = (g.map(model_plot, 'model', 'corr_diff', hue, order=order, palette=palette)
         .set_titles(col_template="{col_name}"))

    if legend:
        g.add_legend(title='Recording', fontsize='medium', markerscale=1.5)
    g.set_xlabels('')
    g.set_ylabels('Corr. Coeff. Difference [%]')
    g.savefig(os.path.join(dataset_dir,  'diff.png'))


if __name__ == '__main__':
    diff_plot()
