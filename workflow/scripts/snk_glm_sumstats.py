import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from glmbf.discrete_x_regression import mle, hessian, log_likelihood
from glmbf import glmbf
import pickle
from tqdm import tqdm

mle_vmap = jax.jit(
    jax.vmap(mle, (0, dict(n=None, X_unique=None, Ty=0), None, None, None)),
    static_argnums=[3, 4],
)
hessian_vmap = jax.jit(
    jax.vmap(hessian, (0, dict(n=None, X_unique=None, Ty=0), None)), static_argnums=[2]
)
ll_vmap = jax.jit(
    jax.vmap(log_likelihood, (0, dict(n=None, X_unique=None, Ty=0), None)),
    static_argnums=[2],
)
compute_stderr_vmap = jax.jit(
    jax.vmap(glmbf.standard_errors, (0, None)), static_argnums=[1]
)


@partial(jax.jit, static_argnames=["glm"])
def compute_summary_stats_multi_y(x, Y, glm):
    sorted_idx = jnp.argsort(x)
    size = jnp.bincount(x.astype(int), length=3)
    cumsize = jnp.cumsum(size)
    indices = jnp.concat([jnp.zeros(1), cumsize])[:-1].astype(int)

    xsorted = x[sorted_idx]
    Ysorted = Y[:, sorted_idx]

    X_unique = jnp.array([[1, 0], [1, 1], [1, 2]]).astype(float)
    Ty = jnp.add.reduceat(glm.suffstat(Ysorted), indices, axis=1)
    n = jnp.add.reduceat(np.ones(x.size), indices)

    summarized_data = dict(n=n, X_unique=X_unique, Ty=Ty)
    b_init = jnp.array([glm.link(Ysorted.mean(1)), jnp.zeros(Ysorted.shape[0])]).T

    Bhat, optstate = mle_vmap(b_init, summarized_data, 0.01, glm, -1)
    gradnorm = jnp.sqrt(np.sum(optstate[2].grad ** 2, 1))
    H = hessian_vmap(Bhat, summarized_data, glm)
    ll0 = ll_vmap(b_init, summarized_data, glm)
    ll = ll_vmap(Bhat, summarized_data, glm)
    llr = ll - ll0
    s2 = compute_stderr_vmap(H, -1)[:, 1]

    res = dict(
        intercept=Bhat[:, 0],
        bhat=Bhat[:, 1],
        s2=s2,
        llr=llr,
        null_intercept=b_init[0, 0],
        ll0=ll0,
        gradnorm=gradnorm,
    )
    return res


def sample_n_individuals(n, haplotypes, ploidy=2):
    k, p = haplotypes.shape
    genotypes = np.zeros((n, p), dtype=np.uint8)
    for _ in range(ploidy):
        genotypes += haplotypes[np.random.choice(k, n, replace=True)]
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


def tree_stack(trees):
    return jax.tree.map(lambda *v: jnp.stack(v), *trees)


if __name__ == "__main__":
    # load haplotypes
    spec = snakemake.params[0]
    print(spec)

    print("Loading haplotypes")
    haplotypes = np.load(snakemake.input.haplotypes)

    # get correct glm module for simulation
    glms = glmbf.get_available_glms()
    glm = glms.get(spec.get("model"))

    print("Sampling genotypes...")
    data = simulate_data(spec, haplotypes, glm)
    X, Y = data["X"], data["y"]

    print("Computing summary statistics...")
    ss = [
        jax.tree.map(np.array, compute_summary_stats_multi_y(x, Y, glm))
        for x in tqdm(X.T)
    ]
    ss = tree_stack(ss)
    pickle.dump(ss, open(snakemake.output.ss, "wb"))
