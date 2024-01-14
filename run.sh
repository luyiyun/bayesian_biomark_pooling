# bayes(或者叫bayes1), --prior_betax flat
# bayes2, --prior_betax normal
# bayes3, --prior_betax normal --prior_a_std 1.0 --prior_b_std 1.0 --prior_beta0_std 1.0

# results6(正常)
# results7(大方差) --sigma2x 10 --sigma2e 10
# results8(小样本) --nKnowX 5
# results9(缺失) --nKnowX 20 20 0 0


##################################################################################################################################################
#
#  sample studies; a, b, sigma2e are generated from a normal distribution
#
##################################################################################################################################################

# FIXME: 这里是错误的，因为sample_studies覆盖了sigma2e的设置
# # ------------- scenario7 -------------
# # - bayes3
# save_root=./results/results7-bayes3
# python ./main.py --tasks all \
#     --prevalence 0.25 0.5 --nrepeat 1000 --ncores 20 --nSamples 100 --nKnowX 10 --sample_studies --n_knowX_balance --prior_sigma_ws inv_gamma \
#     --sigma2x 10 --sigma2e 10 --prior_betax normal --prior_a_std 1.0 --prior_b_std 1.0 --prior_beta0_std 1.0 \
#     --save_root $save_root
# python ./main.py --tasks summarize --summarize_save_fn summ.xlsx --save_root $save_root

# # - bayes3 supp
# save_root=./results/results7-bayes3-supp
# python ./main.py --tasks all \
#     --prevalence 0.25 0.5 --OR 2.75 3.0 --nrepeat 1000 --ncores 20 --nSamples 100 --nKnowX 10 --sample_studies --n_knowX_balance --prior_sigma_ws inv_gamma \
#     --sigma2x 10 --sigma2e 10 --prior_betax normal --prior_a_std 1.0 --prior_b_std 1.0 --prior_beta0_std 1.0 \
#     --save_root $save_root
# python ./main.py --tasks summarize --summarize_save_fn summ.xlsx --save_root $save_root

# # - bayes2 supp
# save_root=./results/results7-bayes2-supp
# python ./main.py --tasks all \
#     --prevalence 0.25 0.5 --OR 2.75 3.0 --nrepeat 1000 --ncores 20 --nSamples 100 --nKnowX 10 --sample_studies --n_knowX_balance --prior_sigma_ws inv_gamma \
#     --sigma2x 10 --sigma2e 10 --prior_betax normal \
#     --save_root $save_root
# python ./main.py --tasks summarize --summarize_save_fn summ.xlsx --save_root $save_root

# # - bayes supp
# save_root=./results/results7-bayes-supp
# python ./main.py --tasks all \
#     --prevalence 0.25 0.5 --OR 2.75 3.0 --nrepeat 1000 --ncores 20 --nSamples 100 --nKnowX 10 --sample_studies --n_knowX_balance --prior_sigma_ws inv_gamma \
#     --sigma2x 10 --sigma2e 10 --prior_betax flat \
#     --save_root $save_root
# python ./main.py --tasks summarize --summarize_save_fn summ.xlsx --save_root $save_root

# # ------------- scenario8 -------------
# # - bayes3
# save_root=./results/results8-bayes3
# python ./main.py --tasks all \
#     --prevalence 0.25 0.5 --OR 1.25 1.5 1.75 2.0 2.25 2.5 2.75 3.0 --nrepeat 1000 --ncores 20 --nSamples 100 --nKnowX 5 --sample_studies --n_knowX_balance \
#     --prior_betax normal --prior_a_std 1.0 --prior_b_std 1.0 --prior_beta0_std 1.0 \
#     --save_root $save_root
# python ./main.py --tasks summarize --summarize_save_fn summ.xlsx --save_root $save_root

# # - bayes2 supp
# save_root=./results/results8-bayes2-supp
# python ./main.py --tasks all \
#     --prevalence 0.25 0.5 --OR 2.75 3.0 --nrepeat 1000 --ncores 20 --nSamples 100 --nKnowX 5 --sample_studies --n_knowX_balance \
#     --prior_betax normal \
#     --save_root $save_root
# python ./main.py --tasks summarize --summarize_save_fn summ.xlsx --save_root $save_root

# # - bayes supp
# save_root=./results/results8-bayes-supp
# python ./main.py --tasks all \
#     --prevalence 0.25 0.5 --OR 2.75 3.0 --nrepeat 1000 --ncores 20 --nSamples 100 --nKnowX 5 --sample_studies --n_knowX_balance \
#     --prior_betax flat \
#     --save_root $save_root
# python ./main.py --tasks summarize --summarize_save_fn summ.xlsx --save_root $save_root

# # ------------- scenario9 -------------
# # - bayes3 supp
# save_root=./results/results9-bayes3-supp
# python ./main.py --tasks all \
#     --prevalence 0.25 0.5 --OR 2.75 3.0 --nrepeat 1000 --ncores 20 --nSamples 100 --nKnowX 20 20 0 0 --sample_studies --n_knowX_balance \
#     --prior_betax normal --prior_a_std 1.0 --prior_b_std 1.0 --prior_beta0_std 1.0 \
#     --save_root $save_root
# python ./main.py --tasks summarize --summarize_save_fn summ.xlsx --save_root $save_root

# # - bayes2 supp
# save_root=./results/results9-bayes2-supp
# python ./main.py --tasks all \
#     --prevalence 0.25 0.5 --OR 2.75 3.0 --nrepeat 1000 --ncores 20 --nSamples 100 --nKnowX 20 20 0 0 --sample_studies --n_knowX_balance \
#     --prior_betax normal \
#     --save_root $save_root
# python ./main.py --tasks summarize --summarize_save_fn summ.xlsx --save_root $save_root

# # - bayes supp
# save_root=./results/results9-bayes-supp
# python ./main.py --tasks all \
#     --prevalence 0.25 0.5 --OR 2.75 3.0 --nrepeat 1000 --ncores 20 --nSamples 100 --nKnowX 20 20 0 0 --sample_studies --n_knowX_balance \
#     --prior_betax flat \
#     --save_root $save_root
# python ./main.py --tasks summarize --summarize_save_fn summ.xlsx --save_root $save_root

# # ------------- scenario6 -------------
# # - bayes2 supp
# save_root=./results/results6-bayes2-supp
# python ./main.py --tasks all \
#     --prevalence 0.25 0.5 --OR 2.75 3.0 --nrepeat 1000 --ncores 20 --nSamples 100 --nKnowX 10 --sample_studies --n_knowX_balance \
#     --prior_betax normal \
#     --save_root $save_root
# python ./main.py --tasks summarize --summarize_save_fn summ.xlsx --save_root $save_root

# - bayes supp
# save_root=./results/results6-bayes-supp
# python ./main.py --tasks all \
#     --prevalence 0.25 0.5 --OR 2.75 3.0 --nrepeat 1000 --ncores 20 --nSamples 100 --nKnowX 10 --sample_studies --n_knowX_balance \
#     --prior_betax flat \
#     --save_root $save_root
# python ./main.py --tasks summarize --summarize_save_fn summ.xlsx --save_root $save_root



##################################################################################################################################################
#
#  a, b, sigma2e are given explicit values
#
##################################################################################################################################################
OTHERS="--prevalence 0.25 0.5 --OR 1.25 1.5 1.75 2.0 2.25 2.5 2.75 3.0 --nrepeat 1000 --ncores 20 --nSamples 100 --n_knowX_balance"

# save_root=./results/results10-bayes
# python ./main.py --tasks all $OTHERS --nKnowX 10 --sigma2x 10 --sigma2e 10 --prior_betax flat --save_root $save_root
# python ./main.py --tasks summarize --summarize_save_fn summ.xlsx --save_root $save_root

# save_root=./results/results10-bayes2
# python ./main.py --tasks all $OTHERS --nKnowX 10 --sigma2x 10 --sigma2e 10 --prior_betax normal --save_root $save_root
# python ./main.py --tasks summarize --summarize_save_fn summ.xlsx --save_root $save_root

# save_root=./results/results10-bayes3
# python ./main.py --tasks all $OTHERS --nKnowX 10 --sigma2x 10 --sigma2e 10 --prior_betax normal --prior_a_std 1.0 --prior_b_std 1.0 --prior_beta0_std 1.0 --save_root $save_root
# python ./main.py --tasks summarize --summarize_save_fn summ.xlsx --save_root $save_root


# save_root=./results/results11
# python ./main.py --tasks all $OTHERS --nKnowX 20 20 0 0 --prior_betax normal --prior_a_std 1.0 --prior_b_std 1.0 --prior_beta0_std 1.0 --save_root ${save_root}-bayes3
# python ./main.py --tasks summarize --summarize_save_fn summ.xlsx --save_root ${save_root}-bayes3

# python ./main.py --tasks all $OTHERS --nKnowX 20 20 0 0 --prior_betax normal --save_root ${save_root}-bayes2
# python ./main.py --tasks summarize --summarize_save_fn summ.xlsx --save_root ${save_root}-bayes2

# python ./main.py --tasks all $OTHERS --nKnowX 20 20 0 0 --prior_betax flat --save_root ${save_root}-bayes
# python ./main.py --tasks summarize --summarize_save_fn summ.xlsx --save_root ${save_root}-bayes


save_root=./results/results12
python ./main.py --tasks all $OTHERS --nKnowX 10 --prior_betax normal --prior_a_std 1.0 --prior_b_std 1.0 --prior_beta0_std 1.0 --save_root ${save_root}-bayes3
python ./main.py --tasks summarize --summarize_save_fn summ.xlsx --save_root ${save_root}-bayes3

python ./main.py --tasks all $OTHERS --nKnowX 10 --prior_betax normal --save_root ${save_root}-bayes2
python ./main.py --tasks summarize --summarize_save_fn summ.xlsx --save_root ${save_root}-bayes2

python ./main.py --tasks all $OTHERS --nKnowX 10 --prior_betax flat --save_root ${save_root}-bayes
python ./main.py --tasks summarize --summarize_save_fn summ.xlsx --save_root ${save_root}-bayes