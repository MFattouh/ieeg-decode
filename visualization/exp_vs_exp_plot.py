import os
import seaborn as sns
import click
import numpy as np
from exp_plot import create_results_df


@click.command()
@click.argument('dataset_dir', type=click.Path(exists=True))
@click.option('-l', '--legend', is_flag=True, help='if passed will add legends outside the plot')
def exp_vs_exp_plot(dataset_dir, legend):
    sns.set_style("darkgrid")
    sns.set_context("notebook", font_scale=1.5)
    results = create_results_df(dataset_dir)
    unique_experiments = set(results.exp)
    for first_exp in unique_experiments:
        other_experiments = unique_experiments - set([first_exp])
        for second_exp in other_experiments:
            corr_tabel = results.loc[(results.exp == first_exp) | (results.exp == second_exp)]\
                .pivot(index='sub', columns='exp', values='corr')
            corr_tabel['sub'] = corr_tabel.index
            hue = 'sub' if legend else None
            g = sns.lmplot(x=first_exp, y=second_exp, hue=hue, size=6, data=corr_tabel, fit_reg=False,
                           legend=legend, legend_out=True)

            g.ax.plot([0, 1], [0, 1], linestyle=':', color='tab:gray')
            g.set(xticks=np.arange(0, 1.2, 0.2).squeeze().tolist(), xlim=(0, 1))
            g.set(yticks=np.arange(0.2, 1.2, 0.2).squeeze().tolist(), ylim=(0, 1))
            g.savefig(os.path.join(dataset_dir,  first_exp + '_vs_' + second_exp + '.png'))


if __name__ == '__main__':
    exp_vs_exp_plot()
