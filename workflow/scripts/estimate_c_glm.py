# The purpose of this notebook is to explore the difference between
# Laplace MLE and ABF for different GLMs
# we consider the case where psi = b0 + b * x
# that is, the model with an intercept and a single covariate
# we consider when the intercept is fixed, and when it is estimated
# in the asymptotic results, this amounts to a different choice in computing the standard error
# either 
#   (1) taking the reciprical of the last diagonal element of the hessian or the negative log likelihood,
#   (2) taking the last diagonal element of minus the inverse hessian.
# We approximate the error through monte-carlo simulation.
#   1. sample (X,Y) \sim P where P is the joint distribution of X and Y (the conditional distribution Y|X determined by the GLM)
#   2. get a montecarlo estimate of E[-1/2 l''(b) * b^2 + l(b) - l(0)

#
#%%
from typing import Callable
from jax import  hessian
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm
import pandas as pd
from jax.scipy.optimize import minimize
from functools import partial
import jax
import jax.scipy.stats.multivariate_normal as mvn

#%% logistic
def logp(y, psi):
    return y * psi - jnp.log(1 + jnp.exp(psi))

def ry(psi):
    return np.random.binomial(1, 1/(1 + np.exp(-psi)))

def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

#%% 
# now we can compute c for a range of distributions on x
# and different observation models

# logistic
def logp_logistic(y, psi):
    return y * psi - jnp.log(1 + jnp.exp(psi))

def sample_y_logistic(psi):
    return np.random.binomial(1, 1/(1 + np.exp(-psi)))

# logistic
def logp_logistic(y, psi):
    return y * psi - jnp.log(1 + jnp.exp(psi))

def sample_y_logistic(psi):
    return np.random.binomial(1, 1/(1 + np.exp(-psi)))

# probit model
def logp_probit(y, psi):
    return y * norm.logcdf(psi) + (1-y) * norm.logcdf(-psi)

def sample_y_probit(psi):
    return np.random.binomial(1, norm.cdf(psi))

# poisson
def logp_poisson(y, psi):
    return y * psi - jnp.exp(psi)

def sample_y_poisson(psi):
    return np.random.poisson(np.exp(psi))


# main function
def estimate_c(logp: Callable, ry: Callable, rx: Callable, b: float, b0: float, n: int):
    x = rx(n)
    y = ry(b0 + b * x) # sample y given linear predictions
    l = lambda beta: jnp.mean(logp(y, b0 + beta * x))
    return dict(
        wald = -hessian(l)(b) * b**2,
        lr = l(b) - l(0),
        b=b,
        b0=b0,
        n=n
    )

# main function
from functools import partial
@partial(jax.jit, static_argnames=['logp'])
def _estimate_c(x, y, b, b0, logp):
    nl = lambda beta:  -jnp.mean(logp(y, b0 + beta * x))
    fit = minimize(nl, jnp.array([b]), method='BFGS')
    bhat = fit.x[0]
    return dict(
        wald = hessian(nl)(bhat) * bhat**2,
        lr = nl(0) - nl(bhat),
        b=b,
        b0=b0,
        n=x.size
    )
def estimate_c(logp: Callable, ry: Callable, rx: Callable, b: float, b0: float, n: int):
    x = rx(n)
    y = ry(b0 + b * x) # sample y given linear predictions
    return _estimate_c(x, y, b, b0, logp)


from functools import partial
@partial(jax.jit, static_argnames=['logp'])
def _estimate_c2d(x, y, b, b0, logp, prior_variance):
    # NOTE: prior variance is a vector of length 2 
    n = x.size
    
    # fit full model
    nl = lambda beta:  -jnp.mean(logp(y, beta[0] + beta[1] * x))
    fit = minimize(nl, jnp.array([b0, b]), method='BFGS')
    
    # fit null model
    nullnl = lambda beta0:  -jnp.mean(logp(y, beta0))
    nullfit = minimize(nullnl, jnp.array([b0]), method='BFGS')
    
    # compute ABF
    s2 = fit.hess_inv[1, 1] / n
    bhat = fit.x[1]
    logabf = norm.logpdf(bhat, 0, jnp.sqrt(s2 + prior_variance[1])) - norm.logpdf(bhat, 0, jnp.sqrt(s2))

    # compute laplace MLE with prior on transformed parameters
    H = hessian(nl)(fit.x)
    w = H[1, 0] / H[0, 0]
    Ainv = jnp.array([[1., -w], [0, 1.]])
    Sigma2 = Ainv @ jnp.diag(prior_variance) @ Ainv.T

    ll = -fit.fun * n
    logp_mle = ll + \
        jnp.log(2 * jnp.pi) - 0.5 * jnp.linalg.slogdet(n*H)[1] + \
        mvn.logpdf(bhat, jnp.zeros(2), Sigma2)

    ll0 = -nullfit.fun * n
    bhat0 = nullfit.x
    s20 = 1 / (n * hessian(nullnl)(bhat0))
    logp0_mle =  0.5 * jnp.log(2 * jnp.pi * s20) + norm.logpdf(bhat0, 0, jnp.sqrt(s20 + prior_variance[0]))
    logbf_lapmle = logp_mle - logp0_mle

    return dict(
        logabf = logabf,
        logbf_lapmle = logbf_lapmle,
        b=b,
        b0=b0,
        n=n
    )

def estimate_c2d(logp: Callable, ry: Callable, rx: Callable, b: float, b0: float, n: int, prior_variance: np.ndarray):
    x = rx(n)
    y = ry(b0 + b * x) # sample y given linear predictions
    return _estimate_c2d(x, y, b, b0, logp, prior_variance)

# %%
def rademacher(n):
    return np.random.binomial(1, 0.5, n) * 2 - 1

def standardized_binomial(n, m=1, p=0.5):
    return (np.random.binomial(m, p, n) - m*p) / np.sqrt(m * p * (1 - p))

def normal(n):
    return np.random.normal(size=n)

# %% 

# when X is rademacher and b0 = 0
# we have a formula for c
# def Elrad(b):
#     b0 = 0
#     a1 = sigmoid(b0 + b) * (b0 + b) - jnp.log(1 + jnp.exp(b0 + b))
#     a2 = sigmoid(b0 - b) * (b0 - b) - jnp.log(1 + jnp.exp(b0 - b))
#     return (a1 + a2) / 2
# b0 = 0.
# b = 2.
# a = estimate_c(logp_logistic, sample_y_logistic, rademacher, b, b0, int(1e6))
# b = 0.5 * sigmoid(b) * sigmoid(-b) * b**2 - Elrad(b) + Elrad(0) 
# a['wald'] - a['lr'] - b # should be close to zero

#%%
if __name__ == '__main__':
    print(snakemake.params.params)

    ry_name = snakemake.params.params['ry']
    ry_kwargs = snakemake.params.get('ry_kwargs', {})

    rx_name = snakemake.params.params['rx']
    rx_kwargs = snakemake.params.params.get('rx_kwargs', {})

    estimate_intercept = snakemake.params.params['estimate_intercept']

    if ry_name == 'logistic':
        ry = sample_y_logistic
        logp = logp_logistic
    elif ry_name == 'probit':
        ry = sample_y_probit
        logp = logp_probit
    elif ry_name == 'poisson':
        ry = sample_y_poisson
        logp = logp_poisson
    
    ry = partial(ry, **ry_kwargs)
    logp = partial(logp, **ry_kwargs)

    if rx_name == 'rademacher':
        rx = rademacher
    elif rx_name == 'normal':
        rx = normal
    elif rx_name == 'standardized_binomial':
        rx = standardized_binomial
    rx = partial(rx, **rx_kwargs)
    
    estimator = estimate_c2d if estimate_intercept else estimate_c

    # standard parameter grid
    bs = np.linspace(-4, 4, 17)
    b0s = np.linspace(-4, 4, 17)
    res = pd.DataFrame([estimator(logp, ry, rx, b, b0, int(1e6)) | ry_kwargs | rx_kwargs for b in bs for b0 in b0s])
    res['pY'] = ry_name
    res['pX'] = rx_name 
    types_to_change = {x: 'float64' for x in ['wald', 'lr', 'b', 'b0']} | {x: 'str' for x in ['pY', 'pX']} | {'n': 'int'}
    
    res = res.astype(types_to_change)
    res.to_feather(snakemake.output[0])