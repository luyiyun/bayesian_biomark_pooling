# bayes(或者叫bayes1), --prior_betax flat
# bayes2, --prior_betax normal
# bayes3, --prior_betax normal --prior_a_std 1.0 --prior_b_std 1.0 --prior_beta0_std 1.0

# results6(正常)
# results7(大方差) --sigma2x 10 --sigma2e 10
# results8(小样本) --nKnowX 5
# results9(缺失) --nKnowX 20 20 0 0


# ------------- scenario7 -------------
# - bayes3
# python ./main.py --tasks all \
#     --prevalence 0.25 0.5 --nrepeat 1000 --ncores 20 --nSamples 100 --nKnowX 10 --sample_studies --n_knowX_balance --prior_sigma_ws inv_gamma \
#     --sigma2x 10 --sigma2e 10 --prior_betax normal --prior_a_std 1.0 --prior_b_std 1.0 --prior_beta0_std 1.0 \
#     --save_root ./results/results7-bayes3
# python ./main.py --tasks summarize --summarize_save_fn summ.xlsx --save_root ./results/results7-bayes3

# - bayes3 supp
python ./main.py --tasks all \
    --prevalence 0.25 0.5 --OR 2.75 3.0 --nrepeat 1000 --ncores 20 --nSamples 100 --nKnowX 10 --sample_studies --n_knowX_balance --prior_sigma_ws inv_gamma \
    --sigma2x 10 --sigma2e 10 --prior_betax normal --prior_a_std 1.0 --prior_b_std 1.0 --prior_beta0_std 1.0 \
    --save_root ./results/results7-bayes3-supp
python ./main.py --tasks summarize --summarize_save_fn summ.xlsx --save_root ./results/results7-bayes3-supp