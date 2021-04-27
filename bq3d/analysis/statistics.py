import sys
self = sys.modules[__name__]

import numpy as np
import pandas as pd
import scipy
from scipy import stats, interpolate
from bq3d import io

import logging

from bq3d._version import __version__
__author__     = 'Ricardo Azevedo, Jack Zeitoun'
__copyright__  = "Copyright 2019, Gandhi Lab"
__license__    = 'BY-NC-SA 4.0'
__version__    = __version__
__maintainer__ = 'Ricardo Azevedo'
__email__      = 'ricardo-re-azevedo@gmail.com'
__status__     = "Development"

log = logging.getLogger(__name__)

def readDataGroup(filenames, combine = True, **args):
    """Turn a list of filenames for data into a np stack"""
    
    # check if stack already an array
    if isinstance(filenames, np.ndarray):
        return filenames

    #read the individual files
    group = []
    for f in filenames:
        data = io.readData(f, **args)
        data = np.reshape(data, (1,) + data.shape)
        group.append(data)

    if combine:
        return np.vstack(group)
    else:
        return group


def readPointsGroup(filenames, **args):
    """Turn a list of filenames for points into a np stack"""
    
    #check if stack already:
    if isinstance(filenames, np.ndarray):
        return filenames

    #read the individual files
    group = []
    for f in filenames:
        data = io.readPoints(f, **args)
        group.append(data)

    return group

def tTestVoxelization(group1, group2, signed = False, removeNaN = True, pcutoff = None):
    """t-Test on differences between the individual voxels in group1 and group2, group is a array of voxelizations"""
    
    g1 = self.readDataGroup(group1)
    g2 = self.readDataGroup(group2)

    tvals, pvals = scipy.stats.ttest_ind(g1, g2, axis = 0, equal_var = True)

    #remove nans
    if removeNaN: 
        pi = np.isnan(pvals)
        pvals[pi] = 1.0
        tvals[pi] = 0

    pvals = self.cutoffPValues(pvals, pcutoff = pcutoff)

    #return
    if signed:
        return pvals, np.sign(tvals)
    else:
        return pvals


def cutoffPValues(pvals, pcutoff = 0.05):
    if pcutoff is None:
        return pvals

    pvals2 = pvals.copy()
    pvals2[pvals2 > pcutoff]  = pcutoff
    return pvals2

def colorPValues(pvals, psign, positive = [1,0], negative = [0,1], pcutoff = None, positivetrend = [0,0,1,0], negativetrend = [0,0,0,1], pmax = None):
    
    pvalsinv = pvals.copy()
    if pmax is None:
        pmax = pvals.max()
    pvalsinv = pmax - pvalsinv

    if pcutoff is None:  # color given p values
        
        d = len(positive)
        ds = pvals.shape + (d,)
        pvc = np.zeros(ds)

        #color
        ids = psign > 0
        pvalsi = pvalsinv[ids]
        pvalsi = pvalsinv[ids]
        for i in range(d):
            pvc[ids, i] = pvalsi * positive[i]

        ids = psign < 0
        pvalsi = pvalsinv[ids]
        for i in range(d):
            pvc[ids, i] = pvalsi * negative[i]

        return pvc

    else:  # split pvalues according to cutoff
    
        d = len(positivetrend)

        if d != len(positive) or  d != len(negative) or  d != len(negativetrend) :
            raise RuntimeError('colorPValues: postive, negative, postivetrend and negativetrend option must be equal length!')

        ds = pvals.shape + (d,)
        pvc = np.zeros(ds)

        idc = pvals < pcutoff
        ids = psign > 0

        ##color 
        # significant postive
        ii = np.logical_and(ids, idc)
        pvalsi = pvalsinv[ii]
        w = positive
        for i in range(d):
            pvc[ii, i] = pvalsi * w[i]

        #non significant postive
        ii = np.logical_and(ids, np.negative(idc))
        pvalsi = pvalsinv[ii]
        w = positivetrend
        for i in range(d):
            pvc[ii, i] = pvalsi * w[i]

        # significant negative
        ii = np.logical_and(np.negative(ids), idc)
        pvalsi = pvalsinv[ii]
        w = negative
        for i in range(d):
            pvc[ii, i] = pvalsi * w[i]

        #non significant postive
        ii = np.logical_and(np.negative(ids), np.negative(idc))
        pvalsi = pvalsinv[ii]
        w = negativetrend
        for i in range(d):
            pvc[ii, i] = pvalsi * w[i]

        return pvc


def mean(group, **args):
    g = readDataGroup(group, **args)
    return g.mean(axis = 0)


def std(group, **args):
    g = readDataGroup(group, **args)
    return g.std(axis = 0)


def var(group, **args):
    g = readDataGroup(group, **args)
    return g.var(axis = 0)


def thresholdPoints(points, intensities, threshold = 0, row = 0):
    """Threshold points by intensities"""
    
    points, intensities = io.readPoints((points, intensities))

    # set up params
    if not isinstance(row, tuple):
        row = (row, row)

    # filter min values
    if intensities.ndim > 1:
        i = intensities[:,row[0]]
    else:
        i = intensities
    iids = np.ones(i.shape, dtype = 'bool')
    if not threshold[0] is all:
        iids = np.logical_and(iids, i >= threshold[0])

    # filter max values
    if intensities.ndim > 1:
        i = intensities[:,row[1]]

    if not threshold[1] is all:
        iids = np.logical_and(iids, i <= threshold[1])

    pointsThresh = points[iids, ...]
    intensitiesThresh = intensities[iids, ...]

    return pointsThresh, intensitiesThresh


def weightsFromPrecentiles(intensities, percentiles = [25,50,75,100]):
    perc = np.percentiles(intensities, percentiles)
    weights = np.zeros(intensities.shape)
    for p in perc:
        ii = intensities > p
        weights[ii] = weights[ii] + 1

    return weights


def tTestPointsInRegions(pointCounts1, pointCounts2, signed = False, removeNaN = True, pcutoff = None, equal_var = False):
    """t-Test on differences in counts of points in labeled regions"""
    
    tvals, pvals = scipy.stats.ttest_ind(pointCounts1, pointCounts2, axis = 1, equal_var = equal_var)

    #remove nans
    if removeNaN: 
        pi = np.isnan(pvals)
        pvals[pi] = 1.0
        tvals[pi] = 0

    pvals = self.cutoffPValues(pvals, pcutoff = pcutoff)

    if signed:
        return pvals, np.sign(tvals)
    else:
        return pvals


######
# DataFrame tests
######

def df_mannwhitney(df, a, b, **kwargs):
    """ get man-whitney p statistic from DataFrame

    Test will be run across rows.

    Arguments:
        df (DataFrame): data to analyze
        a (list): Dataframe columns corresponding to first group
        b (list): Dataframe columns corresponding to second group
        kwargs: arguments to pass to scipy.stats.mannwhitneyu
    Returns:
        DataFrame: p-values by index from df
    """

    sink = pd.DataFrame(index=df.index, columns = ['p'], dtype=np.float)
    for i, r in df.iterrows():
        c = r[a].tolist()
        d = r[b].tolist()
        try:
            u,p = stats.mannwhitneyu(c, d, alternative='two-sided', **kwargs)
        except:
            p = np.nan
        sink['p'][i] = p
    return sink

def df_conditional_negate(df, a, b, negate_col):
    """ negates values if a < b

    Arguments:
        df (DataFrame): data to analyze
        a (str): Dataframe columns corresponding to first group
        b (str): Dataframe columns corresponding to second group
        negate_col (str): column to negate
    Returns:
        DataFrame: negate_col with inverted values
    """

    sink = df.iloc[:,0].copy()
    for i, r in df.iterrows():
        if r[a] < r[b]:
            sink.loc[i] = r[negate_col] * -1
        else:
            sink.loc[i] = r[negate_col]
    return sink


def correctPValues(pvalues, method='BH'):
    """Corrects p-values for multiple testing using various methods

    Arguments:
        pvalues (array): list of p values to be corrected
        method (Optional[str]): method to use: BH = FDR = Benjamini-Hochberg, B = FWER = Bonferoni

    References:
        - `Benjamini Hochberg, 1995 <http://www.jstor.org/stable/2346101?seq=1#page_scan_tab_contents>`_
        - `Bonferoni correction <http://www.tandfonline.com/doi/abs/10.1080/01621459.1961.10482090#.VmHWUHbH6KE>`_
        - `R statistics package <https://www.r-project.org/>`_

    Notes:
        - modified from http://statsmodels.sourceforge.net/ipdirective/generated/scikits.statsmodels.sandbox.stats.multicomp.multipletests.html
    """

    pvals = np.asarray(pvalues)

    if method.lower() in ['bh', 'fdr']:

        pvals_sorted_ids = np.argsort(pvals)
        pvals_sorted = pvals[pvals_sorted_ids]
        sorted_ids_inv = pvals_sorted_ids.argsort()

        n = len(pvals)
        bhfactor = np.arange(1, n + 1) / float(n)

        pvals_corrected_raw = pvals_sorted / bhfactor
        pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
        pvals_corrected[pvals_corrected > 1] = 1

        return pvals_corrected[sorted_ids_inv]

    elif method.lower() in ['b', 'fwer']:

        n = len(pvals)

        pvals_corrected = n * pvals
        pvals_corrected[pvals_corrected > 1] = 1
        return pvals_corrected

def estimateQValues(pvalues, m=None, pi0=None, lowMemory=False):
    """Estimates q-values from p-values

    Arguments:
        pvalues (array): list of p-values
        m (int or None): number of tests. If None, m = pvalues.size
        pi0(float or None): estimate of m_0 / m which is the (true null / total tests) ratio, if None estimation via cubic spline.
        lowMemory (bool): if true use low memory version

    Notes:
        - The q-value of a particular feature can be described as the expected proportion of
          false  positives  among  all  features  as  or  more  extreme  than  the observed one
        - The estimated q-values are increasing in the same order as the p-values

    References:
        - `Storey and Tibshirani, 2003 <http://www.pnas.org/content/100/16/9440.full>`_
        - modified from https://github.com/nfusi/qvalue
    """

    if not (pvalues.min() >= 0 and pvalues.max() <= 1):
        raise RuntimeError("estimateQValues: p-values should be between 0 and 1")

    original_shape = pvalues.shape
    pvalues = pvalues.ravel()  # flattens the array in place, more efficient than flatten()

    if m is None:
        m = float(len(pvalues))
    else:
        # the user has supplied an m
        m *= 1.0

    # if the number of hypotheses is small, just set pi0 to 1
    if len(pvalues) < 100 and pi0 is None:
        pi0 = 1.0
    elif pi0 is not None:
        pi0 = pi0
    else:
        # evaluate pi0 for different lambdas
        pi0 = []
        lam = scipy.arange(0, 0.90, 0.01)
        counts = scipy.array([(pvalues > i).sum() for i in lam])

        for l in range(len(lam)):
            pi0.append(counts[l] / (m * (1 - lam[l])))

        pi0 = scipy.array(pi0)

        # fit natural cubic scipyline
        tck = interpolate.splrep(lam, pi0, k=3)
        pi0 = interpolate.splev(lam[-1], tck)

        if pi0 > 1:
            log.verbose("estimateQValues: got pi0 > 1 (%.3f) while estimating qvalues, setting it to 1" % pi0)
            pi0 = 1.0

    if not (0 <= pi0 <= 1):
        raise RuntimeError("estimateQValues: pi0 is not between 0 and 1: %f" % pi0)

    if lowMemory:
        # low memory version, only uses 1 pvalues and 1 qv matrices
        qv = scipy.zeros((len(pvalues),))
        last_pvalues = pvalues.argmax()
        qv[last_pvalues] = (pi0 * pvalues[last_pvalues] * m) / float(m)
        pvalues[last_pvalues] = -scipy.inf
        prev_qv = last_pvalues
        for i in range(int(len(pvalues)) - 2, -1, -1):
            cur_max = pvalues.argmax()
            qv_i = (pi0 * m * pvalues[cur_max] / float(i + 1))
            pvalues[cur_max] = -scipy.inf
            qv_i1 = prev_qv
            qv[cur_max] = min(qv_i, qv_i1)
            prev_qv = qv[cur_max]

    else:
        p_ordered = scipy.argsort(pvalues)
        pvalues = pvalues[p_ordered]
        qv = pi0 * m / len(pvalues) * pvalues
        qv[-1] = min(qv[-1], 1.0)

        for i in range(len(pvalues) - 2, -1, -1):
            qv[i] = min(pi0 * m * pvalues[i] / (i + 1.0), qv[i + 1])

        # reorder qvalues
        qv_temp = qv.copy()
        qv = scipy.zeros_like(qv)
        qv[p_ordered] = qv_temp

        # reshape qvalues
        qv = qv.reshape(original_shape)

    return qv


def testCramerVonMises2Sample(x, y):
    """
    Computes the Cramer von Mises two sample test.

    This is a two-sided test for the null hypothesis that 2 independent samples
    are drawn from the same continuous distribution.

    Parameters:
        x, y (sequence of 1-D ndarrays): two arrays of sample observations
        assumed to be drawn from a continuous distribution, sample sizes
        can be different

    Returns:
        (float, float): T statistic, two-tailed p-value

    References:
        - modified from https://github.com/scipy/scipy/pull/3659
    """

    # following notation of Anderson et al. doi:10.1214/aoms/1177704477
    N = len(x)
    M = len(y)
    assert N * M * (N + M) < sys.float_info.max

    alldata = np.concatenate((x, y))
    allranks = stats.rankdata(alldata)
    ri = allranks[:N]
    sj = allranks[-M:]
    i = stats.rankdata(x)
    j = stats.rankdata(y)
    # Anderson et al. Eqn 10
    U = N * np.sum((ri - i) ** 2) + M * np.sum((sj - j) ** 2)
    # print U

    # Anderson et al. Eqn 9
    T = U / (N * M * (N + M)) - (4 * M * N - 1) / (6 * (M + N))
    # print T

    Texpected = 1. / 6 + 1. / (6 * (M + N))
    Tvariance = 1. / 45 * (M + N + 1) / (M + N) ** 2 * (4 * M * N * (M + N) - 3 * (M ** 2 + N ** 2) - 2 * M * N) / (
                4 * M * N)

    zscore = np.abs(T - Texpected) / np.sqrt(Tvariance)
    # print zscore

    return T, 2 * scipy.stats.distributions.norm.sf(zscore)

