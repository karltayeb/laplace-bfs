import glmbf.discrete_x_regression as dxr
from glmbf import glmbf
from glmbf import optimization
from jax.scipy.stats import norm
import jax
import jax.numpy as jnp
import numpy as np
import pickle

# vectorized functions for computing BFs
compute_log_abf_vmap = jax.jit(
    jax.vmap(glmbf.compute_log_abf, (0, 0, None)), static_argnums=[2]
)
compute_log_laplacebf_vmap = jax.jit(
    jax.vmap(glmbf.compute_log_laplacebf, (0, 0, 0, None)), static_argnums=[3]
)

# vectorized functions for computing MLEs
summarize_data_vmap = jax.jit(
    jax.vmap(dxr.summarize_data, (0, None, None, None)), static_argnums=[2, 3]
)
compute_mle_vmap = jax.jit(
    jax.vmap(dxr.mle, (None, 0, None, None)), static_argnums=[2, 3]
)
compute_ll_vmap = jax.jit(
    jax.vmap(dxr.log_likelihood, (0, 0, None)), static_argnums=[2]
)
compute_hessian_vmap = jax.jit(jax.vmap(dxr.hessian, (0, 0, None)), static_argnums=[2])
compute_stderr_vmap = jax.jit(
    jax.vmap(glmbf.standard_errors, (0, None)), static_argnums=[1]
)
check_converged_vmap = jax.vmap(
    optimization.check_tol, (0, None)
)  # for checking convergence of mle computation


# functions for creating design matrix
def add_ones(x):
    return jnp.stack([jnp.ones_like(x), x]).T


add_ones_vmap = jax.vmap(add_ones, 1)


def sample_n_individuals(n, haplotypes, ploidy=2):
    K, p = haplotypes.shape
    genotypes = np.zeros((n, p))
    for _ in range(ploidy):
        genotypes = genotypes + haplotypes[np.random.choice(K, n, replace=True)]
    return genotypes


def simulate_data(spec, haplotypes, glm):
    # set random seed
    np.random.seed(spec.get("seed"))

    # sample haplotypes to produce diploid genotypes,
    X = sample_n_individuals(spec.get("sample_size"), haplotypes, 2)

    # sample phenotype
    x = X[:, spec.get("causal_snp")]
    b0 = spec.get("intercept")
    b = spec.get("effect_size")
    y = np.array([glm.sample(b0 + b * x) for _ in range(spec.get("reps"))])
    return dict(X=X, y=y)


def compute_summary_statistics(X, y, prior_variance, glm):
    # prepare data
    Xi = add_ones_vmap(X)
    data = summarize_data_vmap(
        Xi, y, 3, glm
    )  # size=3, there are 3 possible genotypes for diploid

    # compute MLE
    bhat0 = jnp.array([glm.link(y.mean()), 0.0])  # same null fit for all
    Bhat, optstates = compute_mle_vmap(bhat0, data, glm, -1)  # fit MLE
    converged = check_converged_vmap(optstates, 1e-3)

    # compute ll0
    data1 = dxr.summarize_data(
        Xi[0], y, 3, glm
    )  # for evaluating ll0 using summarized data. could be any point
    ll0 = dxr.log_likelihood(bhat0, data1, glm)

    # compute bayes factors, summary statistics
    llr = compute_ll_vmap(Bhat, data, glm) - ll0
    H = compute_hessian_vmap(Bhat, data, glm)
    s2 = compute_stderr_vmap(H, -1)[:, 1]
    z = Bhat[:, -1] / jnp.sqrt(s2)

    res = dict(
        intercept=Bhat[:, 0],
        bhat=Bhat[:, 1],
        s2=s2,
        llr=llr,
        null_intercept=bhat0[0],
        ll0=ll0,
        converged=converged,
    )
    return res


def compute_log_bfs(summary_statistics, prior_variance):
    bhat = summary_statistics.get("bhat")
    s2 = summary_statistics.get("s2")
    z = bhat / jnp.sqrt(s2)
    llr = summary_statistics.get("llr")

    log_abf = norm.logpdf(bhat, 0, jnp.sqrt(s2 + prior_variance)) - norm.logpdf(
        bhat, 0, jnp.sqrt(s2)
    )
    log_laplacebf = log_abf - 0.5 * z**2 + llr

    return dict(
        prior_variance=prior_variance,
        log_abf=log_abf,
        log_laplacebf=log_laplacebf,
    )


compute_ss_vmap = jax.jit(
    jax.vmap(compute_summary_statistics, (None, 0, None, None)), static_argnums=[3]
)

if __name__ == "__main__":
    print(snakemake.rule)

    # load haplotypes
    spec = snakemake.params[0]
    haplotypes = haplotypes = np.load(snakemake.input.haplotypes)

    # get correct glm module for simulation
    glms = glmbf.get_available_glms()
    glm = glms.get(spec.get("model"))

    print("Sampling genotypes...")
    data = simulate_data(spec, haplotypes, glm)
    print("Computing summary statistics...")
    ss = compute_ss_vmap(data["X"], data["y"], 1.0, glm)
    ss = jax.tree.map(np.array, ss)

    pickle.dump(ss, open(snakemake.output.ss, "wb"))
