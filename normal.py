"""
This is a file for trying out implementation of eq 10 and 11 (i.e hierarchical inference steps).

In this, we're only dealing with one variable, "r".
MOreover, the "r" sample we got is from Tanoglidis' BNN output.

We put a sample "r" through hierarchical inference step, implemented in the stan programming
language, and plot it.

"""

import numpy as np
from scipy.stats import norm, uniform
from scipy.special import logsumexp
from chainconsumer import ChainConsumer
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool, cpu_count

# load pystan
import emcee
import random, os, sys
import cProfile
import nest_asyncio
import asyncio

os.environ["OMP_NUM_THREADS"] = "1"
ncpu = cpu_count()

print("{0} CPUS".format(ncpu))

# these samples are obtained through sampling from trained bnn
# in other words: bnn_sampls = xi_k ~ p(xi_k | d_k, Omega_int)
# these are Tanoglidis' uniform priors trained BNN

# note that, if sigma<=0, we reset sigma = uniform(0,5)


def calc_loglik(x, theta, minU, maxU):
    mu, sigma = theta
    if not (minU < mu < maxU):
        mu = np.random.uniform(minU, maxU)
    if (sigma<=0):
        sigma = np.random.uniform(0,5)
    return logsumexp(norm.logpdf(x, mu, sigma)-uniform.logpdf(x, minU, maxU))


def log_likelihood(theta, xi, minU, maxU):
#    # review eq 10 log derivation (summing)
    mu, sigma = theta
    return logsumexp(norm.logpdf(xi, mu, sigma)-uniform.logpdf(xi, minU, maxU))
#     #loglik = []
#     #for x in xi:
#     #    loglik.append(calc_loglik(x, theta, minU, maxU))
#     #    print("log_likelihood calculation theta=", theta)
#
#     #NOTE: check loglik
#       #lp = norm.logpdf(xi, mu, sigma).sum()
#       #lp -= uniform.logpdf(xi, minU, maxU).sum()
#       #return lp
#    print("starting")
#    o = np.sum(np.array([logsumexp(norm.logpdf(x, mu, sigma) - uniform.logpdf(x, minU, maxU)) for x in xi]))
#    print("returning")
#    return o
#    #return logsumexp(norm.logpdf(xi, mu, sigma)-uniform.logpdf(xi, minU, maxU))


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
    print("calculating theta (mu, sigma)=", theta)
    print("and xi=", xi)
    lp = log_prior(theta, minU, maxU)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(theta, xi, minU, maxU)




#--------------------------------------------------------------------------------------------------
# EQ 10
#--------------------------------------------------------------------------------------------------
def eq10(name, inv_path, real_path, minU=2.0, maxU=8.0, ndim=2, nwalkers=8,
                burnin=30, num_samps_from_file =500):
    # load necessary data (r)
    inv_sampl = np.load(str(inv_path))
    y_keep = np.load(str(real_path))

    # Effective radius
    sample_r_eff = inv_sampl[:,:,-1]
    r_eff_true = y_keep[:,-1]
    # NOTE: test set of 100 -1000 galaxies

    #new_sample_r_eff = np.array([np.random.choice(i, 10000) for i in sample_r_eff]).flatten()
    new_sample_r_eff = np.array(sample_r_eff[:num_samps_from_file]).flatten() # TODO: 20?
    # if sample fall outside uniform(minU, maxU), reset it with a sample from uniform(minU, maxU)
    new_sample_r_eff = np.array([np.random.uniform(minU, maxU) if i<minU or i>maxU else i for i in new_sample_r_eff])


    initial_guess = np.array([[7.,7.],[-2,10],[0,0],[9,9]]*2)  # initialize emcee walkers

    pos = initial_guess + 1e-3 * np.random.randn(nwalkers, ndim)
    return new_sample_r_eff, pos


def run_mcmc(new_sample_r_eff, pos, ndim=2, burnin=10, minU=2.0, maxU=8.0, nstep=10000, nwalkers=8):
    #with Pool() as pool:
    print("to run emcee")
    print("working on new_samplr_r_eff=", new_sample_r_eff)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(new_sample_r_eff.astype(np.float32), minU, maxU)) # used to have a = stepsize - which was in the param but not anymore. let's see how this goes`
    print("running mcmc")
    sampler.run_mcmc(pos, burnin, progress=True)
    sampler.reset() # reset after burn in

    sampler.run_mcmc(pos, nsteps, progress=True)


    f = sampler.get_chain(flat=True)
    print("acceptance:", sampler.acceptance_fraction)
    np.save(f"eq10_{name}.npy", f)  # save the data

    print('done eq10------------------------------------------------')
    return f


#if __name__=="__main__":
#    global series_num
#    global new_sample_r_eff
#    inv_samp = str(sys.argv[2])
#    real_samp = str(sys.argv[3])
#    name = str(sys.argv[1])
#    print("going to eq10")
#    new_sample_r_eff, pos = eq10(name, inv_samp, real_samp)
#    run_mcmc(new_sample_r_eff, pos)

#    burnin=1
#    minU=2.0
#    maxU=8.0
#    nstep=10
#    nwalkers=8
#    ndim=2
#    with Pool() as pool:
#        print("to run emcee")
#        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(new_sample_r_eff.astype(np.float32), minU, maxU), pool=pool) # used to have a = stepsize - which was in the param but not anymore. let's see how this goes`
#        print("running mcmc")
#        sampler.run_mcmc(pos, burnin, progress=True)
#        sampler.reset() # reset after burn in
#
#        sampler.run_mcmc(pos, nsteps, progress=True)
#
#
#    f = sampler.get_chain(flat=True)
#    print("acceptance:", sampler.acceptance_fraction)
#    np.save(f"eq10_{name}.npy", f)  # save the data
#
#    print('done eq10------------------------------------------------')





#--------------------------------------------------------------------------------------
#    pool = Pool()
#    print("pooling")
#    pool.apply_async(run_mcmc, args=(new_sample_r_eff, pos))
#    pool.close()
#    pool.join()
#--------------------------------------------------------------------------------------

global series_num
inv_samp = str(sys.argv[2])
real_samp = str(sys.argv[3])
name = str(sys.argv[1])
print("going to eq10")

new_sample_r_eff, pos = eq10(name, inv_samp, real_samp)
minU=2.0
maxU=8.0
ndim=2
nwalkers=8
nsteps=10000
burnin=30

with Pool() as pool:
    print("to ensembleSampler")
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(new_sample_r_eff.astype(np.float32),
                                        minU, maxU), pool=pool) # used to have a = stepsize - which was in the param but not anymore. let's see how this goes`
    print("running mcmc")
    sampler.run_mcmc(pos, burnin, progress=True)
    sampler.reset() # reset after burn in

    sampler.run_mcmc(pos, nsteps, progress=True)


f = sampler.get_chain(flat=True)
print("acceptance:", sampler.acceptance_fraction)
np.save(f"eq10_{name}.npy", f)  # save the data

print('done eq10------------------------------------------------')
