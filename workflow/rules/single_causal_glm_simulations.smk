
import yaml

stdpopsim_params = {
    'chr22_500k': {
        "species_name": "HomSap",
        "contig_name": "chr22",
        "left": 18e6,
        "right": 20e6,
        "model_name": "OutOfAfrica_3G09",
        "sample_sizes": {"CEU": 500_000},
        "min_maf": 0.01
    }
}

rule simulate_haplotypes_stdpopsim:
    params:
        lambda w: stdpopsim_params.get(w.locus)
    output:
      haplotypes = 'output/stdpopsim/{locus}/haplotypes.npy',
      R = 'output/stdpopsim/{locus}/R.npy',
      allele_frequency = 'output/stdpopsim/{locus}/allele_frequency.npy',
      ldscore = 'output/stdpopsim/{locus}/ldscore.npy'
    script: '../scripts/snk_stdpopsim.py'


with open('config/simulations/chr22_single_causal_variable_glm_sim.yaml', 'r') as file:
    simulation_specs = yaml.safe_load(file)

rule simulate_glm_summary_statistics:
    input: 
        haplotypes = lambda w: simulation_specs[w.simid]['haplotypes_path']
    params:
        lambda w: simulation_specs[w.simid]
    output:
        ss = 'output/simulations/{simid}/summary_stats.pkl'
    resources:
        mem_mb = 64000,
        time = "8:0:0"
    script: '../scripts/snk_glm_simulate_sumstats.py'


rule simulate_glm_summary_statistics2:
    input: 
        haplotypes = lambda w: simulation_specs[w.simid]['haplotypes_path']
    params:
        lambda w: simulation_specs[w.simid]
    output:
        ss = 'output/single_variable_selection/{simid}/summary_stats.pkl'
    resources:
        mem_mb = 64000,
        time = "8:0:0"
    script: '../scripts/snk_glm_sumstats.py'

rule run_simulations:
    input: expand("output/single_variable_selection/{simid}/summary_stats.pkl", simid = list(simulation_specs.keys()))


import os 
SIMULATION_DIR = 'output/single_variable_selection/'
SIM_IDS = [s for s in os.listdir(f'{SIMULATION_DIR}') if s.startswith('0x')]
rule aggregate_sumstats:
    params: sim_ids = SIM_IDS
    output: 'output/single_variable_selection/processed_summary_stats.pkl'
    script: '../scripts/aggregate_summary_statistics.py'
