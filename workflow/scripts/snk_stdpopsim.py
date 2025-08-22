import stdpopsim
import numpy as np
import jax.numpy as jnp
import jax
from glmbf import glmbf, logistic, poisson
import matplotlib.pyplot as plt
import seaborn as sns
from glmbf import binary_x_regression


def simulate_haplotypes(
    species_name="HomSap",
    contig_name="chr22",
    left=18e6,
    right=20e6,
    model_name="OutOfAfrica_3G09",
    sample_sizes={"CEU": 10_000},
    min_maf=0.005,
):
    """
    Simulate a matrix of genotypes using stdpopsim and msprime.

    Parameters:
    - species_name: str, the name of the species (e.g., "homo_sapiens").
    - contig_name: str, the name of the contig (e.g., "chr22").
    - model: stdpopsim demographic model, the model to use for simulation.
    - sample_sizes: dict, a dictionary with population names as keys and sample sizes as values (e.g., {"CEU": 1000}).

    Returns:
    - X: numpy array, the matrix of simulated genotypes.
    """
    print("Simulating haplotypes with the following parameters:")
    print(f"  Species Name: {species_name}")
    print(f"  Contig Name: {contig_name}")
    print(f"  Left Position: {left}")
    print(f"  Right Position: {right}")
    print(f"  Model Name: {model_name}")
    print(f"  Sample Sizes: {sample_sizes}")
    print(f"  Minimum MAF: {min_maf}")

    # Get the species
    species = stdpopsim.get_species(species_name)

    # Demographic model and contig
    model = species.get_demographic_model(model_name)
    contig = species.get_contig(
        contig_name, left=left, right=right, mutation_rate=model.mutation_rate
    )

    # Get the simulation engine
    engine = stdpopsim.get_engine("msprime")

    # Simulate the tree sequence
    ts = engine.simulate(model, contig, sample_sizes)

    is_biallelic = lambda v: v.num_alleles == 2

    def is_common(v):
        af = v.genotypes.mean()
        return (min_maf <= af) & (af <= 1 - min_maf)

    haplotypes = np.array(
        [v.genotypes for v in ts.variants() if is_biallelic(v) & is_common(v)]
    ).T
    return haplotypes


def sample_n_individuals(n, haplotypes, ploidy=2):
    K, p = haplotypes.shape
    genotypes = np.zeros((n, p))
    for _ in range(ploidy):
        genotypes = genotypes + haplotypes[np.random.choice(K, n, replace=True)]
    return genotypes


# simulate genotypes
# we'll use haploid genotypes because we can quickly compute the MLE for binary x

if __name__ == "__main__":
    kwargs = snakemake.params[0]
    haplotypes = simulate_haplotypes(**kwargs)
    af = haplotypes.mean(0)
    R = np.corrcoef(haplotypes, rowvar=False)
    ldscore = (R**2).sum(0)

    np.save(snakemake.output.haplotypes, haplotypes.astype(np.uint8))
    np.save(snakemake.output.R, R)
    np.save(snakemake.output.allele_frequency, af)
    np.save(snakemake.output.ldscore, ldscore)
