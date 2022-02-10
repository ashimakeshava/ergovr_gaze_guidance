import pandas as pd
import numpy as np
from scipy import stats


def get_cluster_period(df, threshold=2):
    out = []
    for col in df.columns:
        m = df[col].eq(1)
        g = (df[col] != df[col].shift()).cumsum()[m]
        mask = g.groupby(g).transform('count').ge(threshold)
        filt = g[mask].reset_index()
        output = filt.groupby(col).agg(['first','last'])
        output.insert(0, 'col', col)
        out.append(output)
        tmpdf = pd.concat(out, ignore_index=True)
        tmpdf.columns = ['_'.join(c) for c in tmpdf.columns]
        tmpdf['size'] = tmpdf.time_bin_last - tmpdf.time_bin_first

    return tmpdf


def get_tvalues_main(cond, df, permute=False):
    tmpdf = (
        df
        .groupby(['time_bin', 'subject_id', cond])
        .Proportion
        .mean()
        .reset_index()
    )

    if permute:
        tmpdf['Proportion'] = np.random.permutation(tmpdf['Proportion'].values)

    factors = tmpdf[cond].unique()

    tmpdf = (
        tmpdf
        .groupby(['time_bin', 'subject_id'])
        .apply(lambda df:
               df
               .set_index(cond)
               .pipe(lambda d: 0.5*(d.loc[factors[0], 'Proportion']
               - d.loc[factors[1], 'Proportion']))
              )
        .reset_index(name=cond)
    )

    n_subject = df.subject_id.nunique()
    threshold = -stats.distributions.t.ppf(0.05, n_subject - 1)

    t_val = (
        tmpdf
        .groupby('time_bin')
        [cond]
        .apply(lambda s: pd.Series(
            [np.sqrt(n_subject)*s.mean()/s.std(),
            abs(np.sqrt(n_subject)*s.mean()/s.std())>threshold],
            index=['t_value', 'thresh'])
        )
        .to_frame()
    )
    return t_val

def get_clusters(df, permute=False):
    clusters = pd.concat([
        get_tvalues_main("trial_type", df, permute),
        ]
        , axis=1
    )
    clusters = clusters.reset_index().rename(columns={'level_1':'type'}).set_index('time_bin')

    return clusters

def get_cluster_mass(clusters, cluster_sizes):
    for cl_size in cluster_sizes.iterrows():
        cl_ = clusters.query('type=="t_value"')

        cluster_sizes.loc[cl_size[0], 'mass'] =  (
            cl_.loc[cl_size[1].time_bin_first : cl_size[1].time_bin_last, cl_size[1].col_].sum()
        )

    return cluster_sizes

def simulate_clusters(seed, df):
    np.random.seed(seed)
    clust = get_clusters(df, permute=True)
    cluster_period = get_cluster_period(clust.query('type=="thresh"')[[c for c in clust.columns if c !="type"]])
    rand_cluster = get_cluster_mass(clust, cluster_period.copy())
    return rand_cluster
