# creates a bar plot to compare the average performance per model configuration

import os
import pandas as pd
from glob import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import click
import numpy as np
import scipy.stats
from pathlib import Path


@click.command()
@click.argument('dataset_dir', type=click.Path(exists=True))
@click.option('--kind', '-k', type=click.Choice(['violin', 'box', 'bar']), default='box')
@click.option('-l', '--legend', is_flag=True, help='if passed will add legends outside the plot')
@click.option('--palette', '-p', default='Set2')
@click.option('--test', '-t', type=click.Choice(['pearson', 'wilcoxon']), default='wilcoxon')
def model_plot(dataset_dir, test, kind, legend, palette):
    sns.set_context("notebook", font_scale=1.5)
    sns.set_style("darkgrid")
    results = create_results_df(dataset_dir)
    results.to_csv(os.path.join(dataset_dir, 'all_results.csv'), index=False)
    gdf = results.groupby('model')
    print('mean corr. per model:')
    print(gdf.mean())
    gdf.mean().to_csv(os.path.join(dataset_dir, 'mean_corr.csv'))
    corr_values, pvalues = calculate_pvalues(results, test)
    print('corr. values:')
    print(corr_values)
    print('pvalues:')
    print(pvalues)
    corr_values.to_csv(os.path.join(dataset_dir, test + 'corr_values.csv'))
    pvalues.to_csv(os.path.join(dataset_dir, test + 'pvalues.csv'))
    exp_mode = 'layers' if 'layer' in results.model.values[0].lower() else 'models'
    if exp_mode == 'layers':
        task = Path(dataset_dir).name
        order = sorted(set(results.model), key=lambda model: int(model[0]))
    else:
        task = '_'.join([Path(dataset_dir).name, Path(dataset_dir).parent.name]) 
        #  order = ['HYBRID', 'SHALLOW', 'DEEP4', 'RNN']
        order = ['HYBRID', 'SHALLOW', 'DEEP4', 'DEEP5_x2', 'DEEP5_x4', 'RNN']

    if legend:
        color, bpalette = 'gray', None
    else:
        color, bpalette = 0.3, palette

    fig, ax = plt.subplots(figsize=(10, 6))
    if kind == 'box':
        sns.boxplot('model', 'corr', data=results, color=color, order=order, ax=ax, palette=bpalette)
    elif kind == 'violin':
        sns.violinplot('model', 'corr', data=results, color=color, order=order, ax=ax, palette=bpalette)
    elif kind == 'bar':
        sns.pointplot('model', 'corr', data=results, color=color, join=False, estimator=np.mean, order=order, ci='sd',
                      markers='D', errwidth=1.2, capsize=0.05, palette=bpalette)
    else:
        raise KeyError

    if legend:
        palette = sns.hls_palette(len(set(results['sub'])))
        color = None
        hue = 'sub'
    else:
        color = 0.3
        hue = None
    sns.stripplot('model', 'corr', ax=ax, data=results, hue=hue, jitter=True, color=color, order=order,
                  palette=palette, linewidth=1, edgecolor='gray')
    if legend:
        ax.legend(title='Recording', loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax.set_xlabel('')
    ax.set_ylabel('Corr. Coeff.')
    ax.set_ylim(top=1.0)
    fig.suptitle(f'{task}')
    fig.savefig(os.path.join(dataset_dir, f'{kind.upper()}_{task}_{exp_mode.upper()}.png'),  bbox_inches='tight')


def create_results_df(dataset_dir):
    csv_pathes = glob(os.path.join(dataset_dir, '*/*/*.csv'))
    assert csv_pathes, 'No csv files found!'
    models = [os.path.basename(os.path.dirname(path)) for path in csv_pathes]
    results = pd.DataFrame(columns=['corr', 'model', 'sub'])
    for idx, (model, csv_path) in enumerate(zip(models, csv_pathes)):
        df = pd.read_csv(csv_path)
        subject = os.path.basename(os.path.dirname(os.path.dirname(csv_path)))
        if 'day' not in df.columns.values:
            df.rename(columns={'Unnamed: 0': 'day'}, inplace=True)
        for day in set(df.day.values):
            corr = df.loc[df['day'] == day, 'corr'].mean()
            results = results.append({'corr': corr,
                                      'model': model,
                                      'sub': '_'.join([day.split('_')[0], subject, day.split('_')[-1]])},
                                     ignore_index=True)
    results['corr'] = results['corr'].astype(float)
    return results


def calculate_pvalues(df, test='wilcoxon'):
    columns = set(df['model'])
    dfcols = pd.DataFrame(columns=columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    corr_values = dfcols.transpose().join(dfcols, how='outer')
    for r in columns:
        for c in columns:
            a, b = df.loc[df['model'] == r, 'corr'], df.loc[df['model'] == c, 'corr']
            if test == 'wilcoxon':
                p_val = round(wilcoxon_signed_rank(a, b), 4)
                corr_val = None
            elif test == 'pearson':
                corr_val, p_val = scipy.stats.pearsonr(a, b)
                corr_val = round(corr_val, 4)
                p_val = round(p_val, 4)

            corr_values[r][c] = corr_val
            pvalues[r][c] = p_val

    return corr_values, pvalues


def wilcoxon_signed_rank(a, b):
    """ See http://www.jstor.org/stable/pdf/3001968.pdf?_=1462738643716
    https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test
    Has been validated against R wilcox.test exact variant (with no ties
    atleast), e.g.:
      wilcox.test(c(0,0,0,0,0,0,0,0,0,0,0),
            c(1,2,3,-4,5,6,7,8,-9,10,11),
            paired=TRUE,
           exact=TRUE)
    Ties are handled by using average rank
    Zeros are handled by assigning half of ranksum to
    positive and half to negative
    ->
    p-value = 0.08301"""
    a = np.array(a)
    b = np.array(b)
    assert len(a) == len(b)
    n_samples = len(a)

    diff = a - b
    ranks = scipy.stats.rankdata(np.abs(diff), method='average')
    # unnecessary could have simply used diff in formulas below
    # also...
    signs = np.sign(diff)

    negative_rank_sum = np.sum(ranks * (signs < 0))
    positive_rank_sum = np.sum(ranks * (signs > 0))
    equal_rank_sum = np.sum(ranks * (signs == 0))

    test_statistic = min(negative_rank_sum, positive_rank_sum)
    # add equals half to both sides... so just add half now
    # after taking minimum, reuslts in the same
    test_statistic += equal_rank_sum / 2.0
    # make it more conservative by taking the ceil
    test_statistic = int(np.ceil(test_statistic))

    # apparently I start sum with 1
    # as count_signrank(0,n) is always 1
    # independent of n
    # so below is equivalent to
    # n_as_extreme_sums = 0
    # and using range(0, test-statistic+1)
    n_as_extreme_sums = 1
    for other_sum in range(1, test_statistic+1):
        n_as_extreme_sums += count_signrank(other_sum, n_samples)
    # I guess 2* for twosided test?
    # same as scipy does
    p_val = (2 * n_as_extreme_sums) / (2**float(n_samples))
    return p_val


def count_signrank(k, n):
    """k is the test statistic, n is the number of samples."""
    # ported from here:
    # https://github.com/wch/r-source/blob/e5b21d0397c607883ff25cca379687b86933d730/src/nmath/signrank.c#L84
    u = int(n * (n + 1) / 2)
    c = int(u / 2)
    w = np.zeros(c+1)
    if u < k < 0:
        return 0
    if k > c:
        k = u - k
    if n == 1:
        return 1.
    if w[0] == 1.:
        return w[k]
    w[0] = w[1] = 1.
    for j in range(2, n+1):
        end = min(j*(j+1)//2, c)
        for i in range(end, j-1, -1):
            w[i] += w[i-j]
    return w[k]


if __name__ == '__main__':
    model_plot()
