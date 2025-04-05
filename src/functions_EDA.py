from fredapi import Fred
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.gridspec as gridspec
from collections import OrderedDict
from pandas.plotting import register_matplotlib_converters


from statsmodels.tsa.stattools import adfuller, kpss, bds
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import het_arch
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import skewtest, kurtosistest, skew, kurtosis, boxcox

def detect_cusum(x, threshold=1, drift=0, ending=False, show=True, ax=None):
    """Cumulative sum algorithm (CUSUM) to detect abrupt changes in data.
    Parameters
    ----------
    x : 1D array_like
        data.
    threshold : positive number, optional (default = 1)
        amplitude threshold for the change in the data.
    drift : positive number, optional (default = 0)
        drift term that prevents any change in the absence of change.
    ending : bool, optional (default = False)
        True (1) to estimate when the change ends; False (0) otherwise.
    show : bool, optional (default = True)
        True (1) plots data in matplotlib figure, False (0) don't plot.
    ax : a matplotlib.axes.Axes instance, optional (default = None).
    Returns
    -------
    ta : 1D array_like [indi, indf], int
        alarm time (index of when the change was detected).
    tai : 1D array_like, int
        index of when the change started.
    taf : 1D array_like, int
        index of when the change ended (if `ending` is True).
    amp : 1D array_like, float
        amplitude of changes (if `ending` is True).
    Notes
    -----
    Tuning of the CUSUM algorithm according to Gustafsson (2000)[1]_:
    Start with a very large `threshold`.
    Choose `drift` to one half of the expected change, or adjust `drift` such
    that `g` = 0 more than 50% of the time.
    Then set the `threshold` so the required number of false alarms (this can
    be done automatically) or delay for detection is obtained.
    If faster detection is sought, try to decrease `drift`.
    If fewer false alarms are wanted, try to increase `drift`.
    If there is a subset of the change times that does not make sense,
    try to increase `drift`.
    Note that by default repeated sequential changes, i.e., changes that have
    the same beginning (`tai`) are not deleted because the changes were
    detected by the alarm (`ta`) at different instants. This is how the
    classical CUSUM algorithm operates.
    If you want to delete the repeated sequential changes and keep only the
    beginning of the first sequential change, set the parameter `ending` to
    True. In this case, the index of the ending of the change (`taf`) and the
    amplitude of the change (or of the total amplitude for a repeated
    sequential change) are calculated and only the first change of the repeated
    sequential changes is kept. In this case, it is likely that `ta`, `tai`,
    and `taf` will have less values than when `ending` was set to False.
    See this IPython Notebook [2]_.
    References
    ----------
    .. [1] Gustafsson (2000) Adaptive Filtering and Change Detection.
    .. [2] hhttp://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectCUSUM.ipynb
    Examples
    --------
    >>> from detect_cusum import detect_cusum
    >>> x = np.random.randn(300)/5
    >>> x[100:200] += np.arange(0, 4, 4/100)
    >>> ta, tai, taf, amp = detect_cusum(x, 2, .02, True, True)
    >>> x = np.random.randn(300)
    >>> x[100:200] += 6
    >>> detect_cusum(x, 4, 1.5, True, True)
    >>> x = 2*np.sin(2*np.pi*np.arange(0, 3, .01))
    >>> ta, tai, taf, amp = detect_cusum(x, 1, .05, True, True)
    """

    x = np.atleast_1d(x).astype('float64')
    gp, gn = np.zeros(x.size), np.zeros(x.size)
    ta, tai, taf = np.array([[], [], []], dtype=int)
    tap, tan = 0, 0
    amp = np.array([])
    # Find changes (online form)
    for i in range(1, x.size):
        s = x[i] - x[i-1]
        gp[i] = gp[i-1] + s - drift  # cumulative sum for + change
        gn[i] = gn[i-1] - s - drift  # cumulative sum for - change
        if gp[i] < 0:
            gp[i], tap = 0, i
        if gn[i] < 0:
            gn[i], tan = 0, i
        if gp[i] > threshold or gn[i] > threshold:  # change detected!
            ta = np.append(ta, i)    # alarm index
            tai = np.append(tai, tap if gp[i] > threshold else tan)  # start
            gp[i], gn[i] = 0, 0      # reset alarm
    # THE CLASSICAL CUSUM ALGORITHM ENDS HERE

    # Estimation of when the change ends (offline form)
    if tai.size and ending:
        _, tai2, _, _ = detect_cusum(x[::-1], threshold, drift, show=False)
        taf = x.size - tai2[::-1] - 1
        # Eliminate repeated changes, changes that have the same beginning
        tai, ind = np.unique(tai, return_index=True)
        ta = ta[ind]
        # taf = np.unique(taf, return_index=False)  # corect later
        if tai.size != taf.size:
            if tai.size < taf.size:
                taf = taf[[np.argmax(taf >= i) for i in ta]]
            else:
                ind = [np.argmax(i >= ta[::-1])-1 for i in taf]
                ta = ta[ind]
                tai = tai[ind]
        # Delete intercalated changes (the ending of the change is after
        # the beginning of the next change)
        ind = taf[:-1] - tai[1:] > 0
        if ind.any():
            ta = ta[~np.append(False, ind)]
            tai = tai[~np.append(False, ind)]
            taf = taf[~np.append(ind, False)]
        # Amplitude of changes
        amp = x[taf] - x[tai]

    if show:
        _plot(x, threshold, drift, ending, ax, ta, tai, taf, gp, gn)

    return ta, tai, taf, amp


def _plot(x, threshold, drift, ending, ax, ta, tai, taf, gp, gn):
    """Plot results of the detect_cusum function, see its help."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

        t = range(x.size)
        ax1.plot(t, x, 'b-', lw=2)
        if len(ta):
            ax1.plot(tai, x[tai], '>', mfc='g', mec='g', ms=10,
                     label='Start')
            if ending:
                ax1.plot(taf, x[taf], '<', mfc='g', mec='g', ms=10,
                         label='Ending')
            ax1.plot(ta, x[ta], 'o', mfc='r', mec='r', mew=1, ms=5,
                     label='Alarm')
            ax1.legend(loc='best', framealpha=.5, numpoints=1)
        ax1.set_xlim(-.01*x.size, x.size*1.01-1)
        ax1.set_xlabel('Data #', fontsize=14)
        ax1.set_ylabel('Amplitude', fontsize=14)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax1.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax1.set_title('Time series and detected changes ' +
                      '(threshold= %.3g, drift= %.3g): N changes = %d'
                      % (threshold, drift, len(tai)))
        ax2.plot(t, gp, 'y-', label='+')
        ax2.plot(t, gn, 'm-', label='-')
        ax2.set_xlim(-.01*x.size, x.size*1.01-1)
        ax2.set_xlabel('Data #', fontsize=14)
        ax2.set_ylim(-0.01*threshold, 1.1*threshold)
        ax2.axhline(threshold, color='r')
        ax1.set_ylabel('Amplitude', fontsize=14)
        ax2.set_title('Time series of the cumulative sums of ' +
                      'positive and negative changes')
        ax2.legend(loc='best', framealpha=.5, numpoints=1)
        plt.tight_layout()
        plt.show()

def show_decompose(df, seasonal_periods = 5, title = " "):
    resultAdd = seasonal_decompose(df, model='additive',       period=seasonal_periods)
    resultMul = seasonal_decompose(df, model='multiplicative', period=seasonal_periods)

    # Hodrick-Prescott filter
    # See Ravn and Uhlig: http://home.uchicago.edu/~huhlig/papers/uhlig.ravn.res.2002.pdf
    lamb = 107360000000
    cycleAdd, trendAdd = sm.tsa.filters.hpfilter(resultAdd.trend[resultAdd.trend.notna().values], lamb=lamb)
    cycleMul, trendMul = sm.tsa.filters.hpfilter(resultMul.trend[resultMul.trend.notna().values], lamb=lamb)

    fig = plt.figure(figsize=(15,15), constrained_layout=True)
    spec = gridspec.GridSpec(ncols=4, nrows=5, figure=fig)

    fig.add_subplot(spec[0, :])
    plt.plot(df)

    plt.title(title + ' (Time-Series)')
    # Additive model
    fig.add_subplot(spec[1, :2])
    plt.plot(resultAdd.trend)
    plt.title('Additive Cyclic-Trend')
    fig.add_subplot(spec[2, 0])
    plt.plot(trendAdd)
    plt.xticks([])
    plt.title('Additive Trend component')
    fig.add_subplot(spec[2, 1])
    plt.plot(cycleAdd)
    plt.xticks([])
    plt.title('Additive Cycle component')
    fig.add_subplot(spec[3, :2])
    plt.plot(resultAdd.seasonal)
    plt.title('Additive Seasonal effect')
    fig.add_subplot(spec[4, :2])
    plt.plot(resultAdd.resid)
    plt.title('Additive Residuals')

    # Multiplicative model
    fig.add_subplot(spec[1, 2:])
    plt.plot(resultMul.trend)
    plt.title('Multiplicative Cyclic-Trend')
    fig.add_subplot(spec[2, 2])
    plt.plot(trendMul)
    plt.xticks([])
    plt.title('Multiplicative Trend component')
    fig.add_subplot(spec[2, 3])
    plt.plot(cycleMul)
    plt.xticks([])
    plt.title('Multiplicative Cycle component')
    fig.add_subplot(spec[3, 2:])
    plt.plot(resultMul.seasonal)
    plt.title('Multiplicative Seasonal effect')
    fig.add_subplot(spec[4, 2:])
    plt.plot(resultMul.resid)
    plt.title('Multiplicative Residuals')
    plt.show()

    '''
    Ver as métricas e forças de tendência e sazonalidade:
    '''
    print("Time-Series Level is " + str(round(df.mean(), 2)))
    print("")
    print("Additive Time Series")
    #FtAdd = max(0, 1-np.var(resultAdd.resid)[0]/np.var(resultAdd.trend)[0]); #inicial
    FtAdd = max(0, 1-np.var(resultAdd.resid)/np.var(resultAdd.trend)); # Removed [0]

    print("Strenght of Trend: %.4f" % FtAdd )
    #FsAdd = max(0, 1-np.var(resultAdd.resid)[0]/np.var(resultAdd.seasonal)[0]); #inicial
    FsAdd = max(0, 1-np.var(resultAdd.resid)/np.var(resultAdd.seasonal)); # Removed [0]

    print("Strenght of Seasonality: %.4f" % FsAdd )
    print("")
    print("Multiplicative Time Series")
    #FtMul = max(0, 1-np.var(resultMul.resid)[0]/np.var(resultMul.trend)[0]); #inicial
    FtMul = max(0, 1-np.var(resultMul.resid)/np.var(resultMul.trend)); # Removed [0]

    print("Strenght of Trend: %.4f" % FtMul )
    #FsMul = max(0, 1-np.var(resultMul.resid)[0]/np.var(resultMul.seasonal)[0]); #inicial
    FsMul = max(0, 1-np.var(resultMul.resid)/np.var(resultMul.seasonal)); # Removed [0]

    print("Strenght of Seasonality: %.4f" % FsMul )


def h0_time_series(df, serie = " "):
    print("======================================================\nEstudar normalidade\n======================================================")
    
    # Kurtosis Test
    k, kpval = kurtosistest(df)
    kurtosis_val = kurtosis(df, fisher=True)
    conclusion_kurtosis = "Reject H0: Data is not normal" if kpval[0] < 0.05 else "Fail to reject H0: Data might be normal"
    print(f"Kurtosis Test for {serie}\nStatistic: {k[0]:.4f}\np-value: {kpval[0]:.4f}\nKurtosis value: {kurtosis_val[0]:.4f}\nConclusion: {conclusion_kurtosis}\n------------------------------------------------------")
    
    # Skewness Test
    s, spval = skewtest(df)
    skew_val = skew(df)
    conclusion_skew = "Reject H0: Data is skewed" if spval[0] < 0.05 else "Fail to reject H0: Data might be symmetric"
    print(f"Skew Test for {serie}\nStatistic: {s[0]:.4f}\np-value: {spval[0]:.4f}\nSkewness value: {skew_val[0]:.4f}\nConclusion: {conclusion_skew}\n------------------------------------------------------")
    
    # Jarque-Bera Test
    jb, jbpval= stats.jarque_bera(df)
    conclusion_jb = "Reject H0: Data is not normal" if jbpval < 0.05 else "Fail to reject H0: Data might be normal"
    print(f"Jarque-Bera Test for {serie}\nStatistic: {jb:.4f}\np-value: {jbpval:.4f}\nConclusion: {conclusion_jb}\n------------------------------------------------------")
    
    # Kolmogorov-Smirnov Test
    ks, kspval = stats.kstest(df.values, 'norm')
    conclusion_ks = "Reject H0: Data is not from a normal distribution" if kspval[0] < 0.05 else "Fail to reject H0: Data might be normal"
    print(f"Kolmogorov-Smirnov Test for {serie}\nStatistic: {ks[0]:.4f}\np-value: {kspval[0]:.4f}\nConclusion: {conclusion_ks}\n------------------------------------------------------")
    
    # Engle's ARCH Test
    lm, lmpval, fval, fpval = het_arch(df[df.columns.values[0]].values)
    conclusion_arch = "Reject H0: Heteroscedasticity detected" if lmpval < 0.05 else "Fail to reject H0: No significant heteroscedasticity"
    print(f"Lagrange Multiplier Test for {serie}\nStatistic: {lm:.4f}\np-value: {lmpval:.4f}\nConclusion: {conclusion_arch}\n------------------------------------------------------")
    
    # BDS Test
    result = bds(df[df.columns.values[0]].values, max_dim=6)
    print("Brock Dechert and Scheinkman Test for " + serie)
    print("Dim 2: z-static %.4f Prob %.4f" % (result[0][0], result[1][0]))
    print("Dim 3: z-static %.4f Prob %.4f" % (result[0][1], result[1][1]))
    print("Dim 4: z-static %.4f Prob %.4f" % (result[0][2], result[1][2]))
    print("Dim 5: z-static %.4f Prob %.4f" % (result[0][3], result[1][3]))
    print("Dim 6: z-static %.4f Prob %.4f" % (result[0][4], result[1][4]))

    print("======================================================\nEstudar estacionaridade\n======================================================")

    # Augmented Dickey-Fuller Test
    result = adfuller(df[df.columns.values[0]].values, regression='c')
    print("Augmented Dickey-Fuller Test for " + serie)
    print("Used lags: %d" % result[2])
    print("Num obs: %d" % result[3])
    print("Critical Values:")
    d = OrderedDict(sorted(result[4].items(), key=lambda t: t[1]))
    for key, value in d.items():
        print("\t%s: %.3f" % (key, value))
    conclusion_adf = "Reject H0: The series is stationary" if result[1] < 0.05 else "Fail to reject H0: The series has a unit root (non-stationary)"
    print(f"Augmented Dickey-Fuller Test for {serie}\nADF Statistic: {result[0]:.4f}\np-value: {result[1]:.4f}\nConclusion: {conclusion_adf}\n------------------------------------------------------")
    
    # Kwiatkowski-Phillips-Schmidt-Shin Test
    result = kpss(df[df.columns.values[0]].values, regression='c')
    print("Kwiatkowski-Phillips-Schmidt-Shin Test for " + serie)
    print("KPSS Statistic: %.4f" % result[0])
    print("Critical Values:")
    d = OrderedDict(sorted(result[3].items(), key=lambda t: t[1], reverse=True))
    for key, value in d.items():
        print("\t%s: %.3f" % (key, value));
    critical_values = result[3]
    conclusion_kpss = "Reject H0: The series is not stationary" if result[0] > critical_values['5%'] else "Fail to reject H0: The series is stationary"
    print(f"Kwiatkowski-Phillips-Schmidt-Shin Test for {serie}\nKPSS Statistic: {result[0]:.4f}\nCritical Values: {critical_values}\nConclusion: {conclusion_kpss}\n------------------------------------------------------")

    