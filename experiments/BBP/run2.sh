prevalence=(0.25 0.5)
OR=(1.25 1.5 1.75 2.0 2.25 2.5 2.75 3.0)
nrepeat=100
ncores=10
nSamples="100 100 100 100"
nKnowX="10 10 10 10"

# Run the simulation for different prevalence and odds ratio values with additional covariates
save_root="results/add_covariates"
for prev_i in "${prevalence[@]}"; do
    for OR_i in "${OR[@]}"; do
        python ./simulation.py --prevalence $prev_i --OR $OR_i --nrepeat $nrepeat --ncores $ncores \
            --nSamples $nSamples --n_knowX_balance --nKnowX $nKnowX \
            --solver blackjax --save_root $save_root \
            --betaz 2 3
    done
done


# Half Cauchy prior for sigma
save_root="results/half_cauchy_prior"
for prev_i in "${prevalence[@]}"; do
    for OR_i in "${OR[@]}"; do
        python ./simulation.py --prevalence $prev_i --OR $OR_i --nrepeat $nrepeat --ncores $ncores \
            --nSamples $nSamples --n_knowX_balance --nKnowX $nKnowX \
            --solver blackjax --save_root $save_root \
            --prior_sigma_dist halfcauchy --prior_sigma_args 1.0
    done
done


# imbalanced samples
save_root="results/imbalanced_samples"
for prev_i in "${prevalence[@]}"; do
    for OR_i in "${OR[@]}"; do
        python ./simulation.py --prevalence $prev_i --OR $OR_i --nrepeat $nrepeat --ncores $ncores \
            --nSamples 200 150 100 50 --n_knowX_balance --nKnowX 20 15 10 5 \
            --solver blackjax --save_root $save_root
    done
done