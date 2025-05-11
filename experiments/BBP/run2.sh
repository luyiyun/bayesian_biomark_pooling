set -e # exit on error

prevalence=(0.25 0.5)
OR=(1.25 1.5 1.75 2.0 2.25 2.5 2.75 3.0)
nrepeat=1000
ncores=10
nSamples="100 100 100 100"
nKnowX="10 10 10 10"
results_dir="results/revision3"


# Half Cauchy prior for sigma
# save_root="${results_dir}/half_cauchy_prior"
# for prev_i in "${prevalence[@]}"; do
#     for OR_i in "${OR[@]}"; do
#         python ./simulation.py --prevalence $prev_i --OR $OR_i --nrepeat $nrepeat --ncores $ncores \
#             --nSamples $nSamples --n_knowX_balance --nKnowX $nKnowX \
#             --solver blackjax --save_root $save_root \
#             --prior_sigma_dist halfcauchy --prior_sigma_args 1.0
#     done
# done


# imbalanced samples
# save_root="${results_dir}/imbalanced_samples"
# for prev_i in "${prevalence[@]}"; do
#     for OR_i in "${OR[@]}"; do
#         python ./simulation.py --prevalence $prev_i --OR $OR_i --nrepeat $nrepeat --ncores $ncores \
#             --nSamples 200 150 100 50 --n_knowX_balance --nKnowX 20 15 10 5 \
#             --solver blackjax --save_root $save_root
#     done
# done


# Run the simulation for different prevalence and odds ratio values with additional covariates
# for n_cov in 1 2 3 4 5; do
#     echo "Running with $n_cov covariates"
#     save_root="${results_dir}/add_covariates${n_cov}"

#     betaz=""
#     for ((i=0; i<${n_cov}; i++)); do
#         betaz+="1 "
#     done

#     for prev_i in "${prevalence[@]}"; do
#         for OR_i in "${OR[@]}"; do
#             python ./simulation.py --prevalence $prev_i --OR $OR_i --nrepeat $nrepeat --ncores $ncores \
#                 --nSamples $nSamples --n_knowX_balance --nKnowX $nKnowX \
#                 --solver blackjax --save_root $save_root \
#                 --betaz $betaz
#         done
#     done
# done

# Run the simulation for prevalance 0.05
# save_root="${results_dir}/prevalence0.05"
# for OR_i in "${OR[@]}"; do
#     python ./simulation.py --prevalence 0.05 --OR $OR_i --nrepeat $nrepeat --ncores $ncores \
#         --nSamples $nSamples --n_knowX_balance --nKnowX $nKnowX \
#         --solver blackjax --save_root $save_root
# done

# Run the simulation to test the running time for different sample sizes
# sample_sizes=(100 300 500 700 900)
# for sample_i in "${sample_sizes[@]}"; do
#     save_root="${results_dir}/sample_size${sample_i}"
#     n_know_X=$((sample_i / 10))
#     python ./simulation.py --prevalence 0.5 --OR 1.5 --nrepeat $nrepeat --ncores $ncores \
#         --nSamples $sample_i $sample_i $sample_i $sample_i \
#         --n_knowX_balance --nKnowX $n_know_X $n_know_X $n_know_X $n_know_X \
#         --solver blackjax --save_root $save_root
# done


# Run the simulation to test the non-informative prior vs informative prior
save_root="${results_dir}/informative_prior"
for prev_i in "${prevalence[@]}"; do
    for OR_i in "${OR[@]}"; do
        python ./simulation.py --prevalence $prev_i --OR $OR_i --nrepeat $nrepeat --ncores $ncores \
            --nSamples $nSamples --n_knowX_balance --nKnowX $nKnowX \
            --solver blackjax --save_root $save_root \
            --prior_x informative
    done
done