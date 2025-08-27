import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from glmbf.discrete_x_regression import mle, hessian, log_likelihood
from glmbf.dx_multiple_y import compute_summary_stats_multiple_y

from glmbf import glmbf
import pickle
from tqdm import tqdm
from util import dmap


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
        compute_summary_stats_multiple_y(x, Y, 3, penalty=0.001, glm=glm)
        for x in tqdm(X.T)
    ]
    ss = tree_stack(ss)
    ss = dmap(lambda x: np.array(x), ss)
    pickle.dump(ss, open(snakemake.output.ss, "wb"))
