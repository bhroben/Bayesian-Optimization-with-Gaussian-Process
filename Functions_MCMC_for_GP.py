import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.stats import multivariate_normal
import itertools
from plotly import graph_objs as go
from scipy.stats import gaussian_kde
import pandas as pd

#fix random seed
np.random.seed(10)

def kernel_M52(x1, x2, l=None, s=1):
    """
    Matern 5/2 kernel

    Parameters
    ----------
    x1 : array_like
        First input
    x2 : array_like
        Second input
    l : array_like 
        Length scales
    s : float
        Signal variance
    """

    # initialize length scale if not provided
    if l is None:
        l = np.ones(len(x1))

    # check if l and x have the same dimension
    if np.isscalar(l):
        if len(x1) != 1:
            raise ValueError('Dimension of x and l must be the same')
    elif len(x1) != len(l):
        raise ValueError('Dimension of x and l must be the same')
    r2 = np.sum(((x1-x2)/l)**2)
    return np.abs(s)*(1 + np.sqrt(5*r2) + 5/3*r2)*np.exp(-np.sqrt(5*r2))

def prior(theta,bounds):
    
    #check bound dimensions
    if len(bounds)!=len(theta):
        raise ValueError('Dimension of bounds and theta must be the same')
    
    for i in range(len(theta)):
        if i==0:
            if (theta[i]<bounds[0][0]) or (theta[i]>bounds[0][1]):
                return 0
        else:
            if (theta[i]>bounds[i][1]) or (theta[i]<bounds[i][0]):
                return 0
            
    return 1

def Metropolis(n_sample, cov, x0, burn_in, thinning, target, **kwargs):
    """
    Metropolis algorithm with Gaussian proposal distribution

    Parameters
    ----------
    target : function
        Target distribution
    x0 : array_like
        Initial point
    n_sample : int
        Number of samples
    cov : array_like
        Covariance matrix of the proposal distribution
    burn_in : int
        Number of burn-in samples
    thinning : int
        Thinning factor
    **kwargs :
        Additional parameters for the target distribution

    Returns
    -------
    array_like
        Samples from the target distribution
    """
    # rows: dimensions, columns: samples (take into account thinning)
    sample = np.zeros((len(x0), n_sample))
    x_current = x0
    acceptance_rate = 0
    #append the chain samples to a list
    all_samples = []
    all_samples.append(x_current)
    for i in range(burn_in):
        # propose new sample
        proposal = multivariate_normal(mean=x_current, cov=cov)
        x_prop = proposal.rvs()
        #s=np.abs(x_prop[0])
        #x_prop[0]=s
        # Acceptance probability
        A = min(1, target(x_prop, **kwargs) / target(x_current, **kwargs))
        if np.random.rand() < A:
            acceptance_rate += 1
            x_current = x_prop
        
        all_samples.append(x_current)
    #calculate acceptance rate
    for i in range(n_sample*thinning):
        # propose new sample
        proposal = multivariate_normal(mean=x_current, cov=cov)
        #extract random sample from the proposal distribution
        x_prop = proposal.rvs()
        #s=np.abs(x_prop[0])
        #x_prop[0]=s
        # Acceptance probability
        A = min(1, target(x_prop, **kwargs) / target(x_current, **kwargs))
        if np.random.rand() < A:
            acceptance_rate += 1
            x_current = x_prop
        all_samples.append(x_current)

        #if i is a multiple of the thinning factor, store the sample
        if i % thinning == 0:
            #store the sample in the 
            sample[:, i//thinning] = x_current
            #print likelihood every 100 samples
            #if i%1000==0:
                ##print('likelihood:', target(x_current, **kwargs))

    print('Acceptance rate: ', acceptance_rate/(n_sample*thinning+burn_in))
    
    return sample,all_samples

def Metropolis_vihola(n_sample, cov0, x0, burn_in, thinning,target, learnin_rate,alpha, **kwargs):
    """
    Metropolis algorithm with Gaussian proposal distribution

    Parameters
    ----------
    target : function
        Target distribution
    x0 : array_like
        Initial point
    n_sample : int
        Number of samples
    cov : array_like
        Covariance matrix of the proposal distribution
    burn_in : int
        Number of burn-in samples
    thinning : int
        Thinning factor
    **kwargs :
        Additional parameters for the target distribution

    Returns
    -------
    array_like
        Samples from the target distribution
    """
    # rows: dimensions, columns: samples (take into account thinning)
    sample = np.zeros((len(x0), n_sample))
    x_current = x0
    acceptance_rate = 0
    #append the chain samples to a list
    all_samples = []
    all_samples.append(x_current)
    cov=cov0
    for i in range(burn_in):

        #decompose the covariance matrix with the Cholesky decomposition
        L = np.linalg.cholesky(cov)
        #generate a random sample from the standard normal distribution
        un = multivariate_normal(mean=np.zeros(len(x0)), cov=np.eye(len(x0))).rvs()
        #make a proposal
        x_prop = x_current + L @ un


        # Acceptance probability
        A = min(1, target(x_prop, **kwargs) / target(x_current, **kwargs))
        if np.random.rand() < A:
            acceptance_rate += 1
            x_current = x_prop
        
        all_samples.append(x_current)
        #update the covariance matrix
        cov= L @ (np.eye(len(x0))+learnin_rate*(A-alpha)*np.outer(un,un)/np.linalg.norm(un)**2) @ L.T

    #calculate acceptance rate
    for i in range(n_sample*thinning):

        #decompose the covariance matrix with the Cholesky decomposition
        L = np.linalg.cholesky(cov)
        #generate a random sample from the standard normal distribution
        un = multivariate_normal(mean=np.zeros(len(x0)), cov=np.eye(len(x0))).rvs()
        #make a proposal
        x_prop = x_current + L @ un


        # Acceptance probability
        A = min(1, target(x_prop, **kwargs) / target(x_current, **kwargs))
        if np.random.rand() < A:
            acceptance_rate += 1
            x_current = x_prop
        all_samples.append(x_current)

        #update the covariance matrix
        cov= L @ (np.eye(len(x0))+learnin_rate*(A-alpha)*np.outer(un,un)/np.linalg.norm(un)**2) @ L.T

        #if i is a multiple of the thinning factor, store the sample
        if i % thinning == 0:
            #store the sample in the 
            sample[:, i//thinning] = x_current
            #print likelihood every 100 samples
            #if i%1000==0:
                ##print('likelihood:', target(x_current, **kwargs))

    print('Acceptance rate: ', acceptance_rate/(n_sample*thinning+burn_in))
    
    return sample,all_samples


def compute_kde(samples, x_range=None):
    if x_range is None:
        x_range = min(samples), max(samples)

    g_kde = gaussian_kde(samples)
    x_kde = np.linspace(*x_range, num=100)

    return x_kde, g_kde(x_kde)

def plot_chain(
        samples, burnin=0.2, initial=0.01, nsig=1, fmt='-', y_range=None,
        width=1000, height=400, margins={'l':20, 'r':20, 't':50, 'b':20}):
    
    plasma = [
        'rgb(13, 8, 135, 1.0)',
        'rgb(70, 3, 159, 1.0)',
        'rgb(114, 1, 168, 1.0)',
        'rgb(156, 23, 158, 1.0)',
        'rgb(189, 55, 134, 1.0)',
        'rgb(216, 87, 107, 1.0)',
        'rgb(237, 121, 83, 1.0)',
        'rgb(251, 159, 58, 1.0)',
        'rgb(253, 202, 38, 1.0)',
        'rgb(240, 249, 33, 1.0)'
    ]

    num_samples = len(samples)
    
    idx_burnin = int(num_samples*burnin)
    idx_initial = int(num_samples*initial) + 1

    sample_steps = np.arange(num_samples)
    
    window = int(0.2*num_samples)
    df = pd.DataFrame(samples, columns=['samples'])
    df['low_q'] = df['samples'].rolling(window=window, center=True, min_periods=0).quantile(quantile=0.05)
    df['high_q'] = df['samples'].rolling(window=window, center=True, min_periods=0).quantile(quantile=0.95)
    
    estimate = np.mean(samples)
    stddev = np.std(samples)
    title = f'The estimate over the chain is: {estimate:0.2f} ± {stddev:0.2f}'

    samples_posterior = samples[idx_burnin:]
    samples_burnin = samples[:idx_burnin]
    samples_initial = samples[:idx_initial]

    if y_range is None:
        std_post = np.std(samples_posterior)
        y_range = min(samples) - nsig * std_post, max(samples) + nsig * std_post

    x_kde_posterior, y_kde_posterior = compute_kde(samples_posterior)
    x_kde_burnin, y_kde_burnin = compute_kde(samples_burnin, x_range=y_range)
    x_kde_initial, y_kde_initial = compute_kde(samples_initial, x_range=y_range)
    
    kde_trace_posterior = go.Scatter(
        x=y_kde_posterior,
        y=x_kde_posterior,
        mode = 'lines',
        line = {
            'color': plasma[4],
            'width': 2
        },
        name='Posterior Distribution',
        xaxis="x2",
        yaxis="y2",
        fill="tozerox",
        fillcolor='rgba(100, 0, 100, 0.20)',
    )
    
    kde_trace_burnin = go.Scatter(
        x=y_kde_burnin,
        y=x_kde_burnin,
        mode = 'lines',
        line = {
            'color': plasma[6],
            'width': 2
        },
        name='Burnin Distribution',
        xaxis="x2",
        yaxis="y2",
        fill="tozerox",
        fillcolor='rgba(100, 0, 100, 0.20)',
    )

    kde_trace_initial = go.Scatter(
        x=y_kde_initial,
        y=x_kde_initial,
        mode = 'lines',
        line = {
            'color': plasma[1],
            'width': 2
        },
        name='Initial Distribution',
        xaxis="x2",
        yaxis="y2",
        fill="tozerox",
        fillcolor='rgba(100, 0, 100, 0.20)',
    )

    plots = [
        kde_trace_initial,
        kde_trace_burnin,
        kde_trace_posterior,
        go.Scatter(
            x=sample_steps,
            y=df['low_q'],
            line={'color':'rgba(255, 0, 0, 0.0)'},
            showlegend=False
        ),
        
        # fill between the endpoints of this trace and the endpoints of the trace before it
        go.Scatter(
            x=sample_steps,
            y=df['high_q'],
            line={'color':'rgba(255, 0, 0, 0.0)'},
            fill="tonextx",
            fillcolor='rgba(100, 0, 100, 0.20)',
            name='Quantile 1 - 99% Region'
        ),
        go.Scatter(
            x=sample_steps[idx_burnin:],
            y=samples_posterior,
            name='Posterior Distribution',
            line={'color':plasma[4]}
        ),
        go.Scatter(
            x=sample_steps[:idx_burnin],
            y=samples_burnin,
            name='Burn-in Region',
            line={'color':plasma[6]}
        ),
        go.Scatter(
            x=sample_steps[:idx_initial],
            y=samples_initial,
            name='Initial Condition Dominated Region',
            line={'color':plasma[1]}
        )
    ]
    
    layout = go.Layout(
        title=title,
        xaxis={'domain': [0, 0.88], 'showgrid': False},
        xaxis2={'domain': [0.9, 1], 'showgrid': False},
        yaxis={'range':y_range, 'showgrid': False},
        yaxis2={
            'anchor': 'x2',
            'range': y_range,
            'showgrid': False
        },
        width=width,
        height=height,
        margin=margins,
        plot_bgcolor='rgba(255, 255, 255, 1)',
        paper_bgcolor='rgba(255, 255, 255, 1)'
    )

    fig = go.Figure(plots, layout=layout)
    fig.show()
