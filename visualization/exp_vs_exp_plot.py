import os
import seaborn as sns
import click
import numpy as np
from exp_plot import create_results_df


@click.command()
@click.argument('dataset_dir', type=click.Path(exists=True))
@click.option('-l', '--legend', is_flag=True, help='if passed will add legends outside the plot')
def model_vs_model_plot(dataset_dir, legend):
    sns.set_style("darkgrid")
    sns.set_context("notebook", font_scale=1.5)
    results = create_results_df(dataset_dir)
    unique_models = set(results.model)
    for first_model in unique_models:
        other_models = unique_models - set([first_model])
        for second_model in other_models:
            corr_tabel = results.loc[(results.model == first_model) | (results.model == second_model)]\
                .pivot(index='sub', columns='model', values='corr')
            corr_tabel['sub'] = corr_tabel.index
            hue = 'sub' if legend else None
            palette = sns.hls_palette(len(set(corr_tabel['sub']))) if legend else None

            g = sns.lmplot(x=first_model, y=second_model, hue=hue, size=6, data=corr_tabel, fit_reg=False,
                           legend=False, palette=palette, scatter_kws={'linewidths': 1, 'edgecolors': 'gray'})

            g.ax.plot([0, 1], [0, 1], linestyle=':', color='tab:gray')
            g.set(xticks=np.arange(0, 1.2, 0.2).squeeze().tolist(), xlim=(0, 1))
            g.set(yticks=np.arange(0.2, 1.2, 0.2).squeeze().tolist(), ylim=(0, 1))
            if legend:
                g.ax.legend(title='Recording', loc='center', bbox_to_anchor=(1.25, 0.5))
            g.savefig(os.path.join(dataset_dir,  first_model + '_vs_' + second_model + '.png'))


if __name__ == '__main__':
    model_vs_model_plot()
