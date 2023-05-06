"""
This is a file for trying out implementation of eq 10 and 11 (i.e hierarchical inference steps).

In this, we're only dealing with one variable, "r".
MOreover, the "r" sample we got is from Tanoglidis' BNN output.

We put a sample "r" through hierarchical inference step, implemented in the stan programming
language, and plot it.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import logsumexp
import nest_asyncio
import asyncio
from chainconsumer import ChainConsumer
from sklearn.preprocessing import StandardScaler
import sys
import multiprocessing

# load pystan
import stan
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import random

# these samples are obtained through sampling from trained bnn
# in other words: bnn_sampls = xi_k ~ p(xi_k | d_k, Omega_int)
# these are Tanoglidis' uniform priors trained BNN

# note that, if sigma<=0, we reset sigma = uniform(0,5)

import numpy as np
import emcee
from scipy.stats import norm, uniform
from scipy.special import logsumexp


def calc_loglik(x, theta, minU, maxU):
    mu, sigma = theta
    if not (minU < mu < maxU):
        mu = np.random.uniform(minU, maxU)
    if (sigma<=0):
        sigma = np.random.uniform(0,5)
    return logsumexp(norm.logpdf(x, mu, sigma)-uniform.logpdf(x, minU, maxU))


def eq10(name, inv_path, real_path, minU=2.0, maxU=8.0, ndim=2, nwalkers=4, nsteps=100, nsamples=1000, stepsize=5, burnin=100):
    # load necessary data (r)
    inv_sampl = np.load(str(inv_path))
    y_keep = np.load(str(real_path))

    # Effective radius
    sample_r_eff = inv_sampl[:,:,-1]
    r_eff_true = y_keep[:,-1]
    # NOTE: test set of 100 -1000 galaxies

    #new_sample_r_eff = np.array([np.random.choice(i, 10000) for i in sample_r_eff]).flatten()
    new_sample_r_eff = np.array(sample_r_eff[:20]).flatten()
    # if sample fall outside uniform(minU, maxU), reset it with a sample from uniform(minU, maxU)
    new_sample_r_eff = np.array([np.random.uniform(minU, maxU) if i<minU or i>maxU else i for i in new_sample_r_eff])


    def log_likelihood(theta, xi, minU, maxU):
        # review eq 10 log derivation (summing)
        #return logsumexp(norm.logpdf(xi, mu, sigma)-uniform.logpdf(xi, minU, maxU))
        loglik = []
        for x in xi:
            loglik.append(calc_loglik(x, theta, minU, maxU))

        # NOTE: check loglik
            #lp = norm.logpdf(xi, mu, sigma).sum()
            #lp -= uniform.logpdf(xi, minU, maxU).sum()
            #return lp
        return np.sum(np.array(loglik))
        #return logsumexp(norm.logpdf(xi, mu, sigma)-uniform.logpdf(xi, minU, maxU))

    def log_prior(theta, minU, maxU):
        # log p(Omega)
        mu, sigma = theta
        if sigma<=0:
            return -np.inf
        lp = norm.logpdf(mu, sigma)
        if not np.isfinite(lp):
            return -np.inf
        return lp


    def log_posterior(theta, xi, minU, maxU):
        # to see whether theta actually has mu and sigma in it
        print("calculating theta (mu, sigma)=", theta, "and xi=", xi)
        lp = log_prior(theta, minU, maxU)
        if np.isinf(lp):
            return -np.inf
        return lp + log_likelihood(theta, xi, minU, maxU)

    initial_guess = np.array([[7.,7.],[-2,10],[0,0],[9,9]])  # initialize emcee walkers

    pos = initial_guess + 1e-3 * np.random.randn(nwalkers, ndim)
    #nsteps = 10000
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(new_sample_r_eff.astype(np.float32), minU, maxU), a=stepsize)
    print("running mcmc")
    sampler.run_mcmc(pos, burnin, progress=True)
    sampler.reset() # reset after burn in

    sampler.run_mcmc(pos, nsteps, progress=True)

    f = sampler.get_chain(flat=True)
    print("acceptance:", sampler.acceptance_fraction)
    np.save(f"eq10_{name}.npy", f)  # save the data

    print('done eq10------------------------------------------------')
    return f



def eq11(data=None, mean_BNN=0, sigma_BNN=1):
    # begin of stan code
    data = pd.read_pickle('eq10_50.pkl')


    norm_data = {
        'N': len(data),
        'mu': np.array(data['mu']),
        'sigma': np.array(data['sigma'])
        }

    norm_code="""
    data {
        int<lower=1> N;
        real mu[N];
        real sigma[N];
    }

    parameters {
        real<lower=2.5, upper=6> xi;
    }

    model {
        // Priors
        xi ~ normal(3.2, 0.36); // prior for xi_k, is log_BNN, or p(xi|d, Omega)
        // note that mean and std of BNN is manually calculated from BNN output

        real p[N];
            for (i in 1:N) {
                // exp(log p(xi | Omega) - log p(xi | Omega_int))
                p[i] = exp(normal_lpdf(xi | mu[i], sigma[i]) - uniform_lpdf(xi | 2.5, 6));
        }

        target += log(sum(p)); // eq 11

    }

    """

    posterior = stan.build(norm_code, data=norm_data)

    # sampling 120300 samples
    fit = posterior.sample(num_chains=10, num_samples=10000)

    f = fit.to_frame()

    f.to_pickle(f'eq11_{series_num}.pkl')

    print('done eq11------------------------------------------------')
    return f


if __name__=="__main__":
    global series_num
    inv_samp = str(sys.argv[2])
    real_samp = str(sys.argv[3])
    name = str(sys.argv[1])
    data = eq10(name, inv_samp, real_samp)
    #eq11(data, m, s)
    #eq11()
