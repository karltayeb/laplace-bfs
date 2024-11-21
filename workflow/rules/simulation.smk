import yaml
# rule simulate_logistic_binary_x:
#     output: 'results/simulations/{sim_id}.pkl'
#     params: 
#         n= 1000,
#         b= 1.,
#         b0= -1,
#         p= 0.1
#     script: '../scripts/simulate_binary_x.py'

# rule compute_bfs:
#     output: 'results/simulations/{sim_id}.fea'
#     params: 
#         params = lambda w: simulation_params[w.sim_id] 
#     script: '../scripts/compute_bfs.py'

rule simulation_one:
    output: 'results/simulations/sim1.fea'
    script: '../scripts/simulation1.py'

# run for
# (logistic, probit, poisson)
# (rademacher, binomial, normal)
# here we write the parameter dictionary
estimate_c_config = yaml.safe_load(open('workflow/resources/estimate_c.yaml', 'r'))
print(estimate_c_config)
rule estimate_c_glm:
    output: 'results/simulations/estimate_c/{id}.fea'
    params: params = lambda w: estimate_c_config.get(w.id)
    script: '../scripts/estimate_c_glm.py'

rule estimate_c_glm_all:
    input:
        expand('results/simulations/estimate_c/{id}.fea', id = estimate_c_config.keys())
        
    output:
        'results/simulations/estimate_c.txt'
    shell:
        """
        touch {output}
        """