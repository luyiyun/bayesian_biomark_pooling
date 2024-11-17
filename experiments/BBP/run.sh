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
# OTHERS="--prevalence 0.25 0.5 --OR 1.25 1.5 1.75 2.0 2.25 2.5 2.75 3.0 --nrepeat 1000 --ncores 20 --nSamples 100 --n_knowX_balance"

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


# save_root=./results/results12
# python ./main.py --tasks all $OTHERS --nKnowX 10 --prior_betax normal --prior_a_std 1.0 --prior_b_std 1.0 --prior_beta0_std 1.0 --save_root ${save_root}-bayes3
# python ./main.py --tasks summarize --summarize_save_fn summ.xlsx --save_root ${save_root}-bayes3

# python ./main.py --tasks all $OTHERS --nKnowX 10 --prior_betax normal --save_root ${save_root}-bayes2
# python ./main.py --tasks summarize --summarize_save_fn summ.xlsx --save_root ${save_root}-bayes2

# python ./main.py --tasks all $OTHERS --nKnowX 10 --prior_betax flat --save_root ${save_root}-bayes
# python ./main.py --tasks summarize --summarize_save_fn summ.xlsx --save_root ${save_root}-bayes


# save_root=./results/results13
# python ./main.py --tasks all $OTHERS --nKnowX 5 --prior_betax normal --prior_a_std 1.0 --prior_b_std 1.0 --prior_beta0_std 1.0 --save_root ${save_root}-bayes3
# python ./main.py --tasks summarize --summarize_save_fn summ.xlsx --save_root ${save_root}-bayes3

# python ./main.py --tasks all $OTHERS --nKnowX 5 --prior_betax normal --save_root ${save_root}-bayes2
# python ./main.py --tasks summarize --summarize_save_fn summ.xlsx --save_root ${save_root}-bayes2

# python ./main.py --tasks all $OTHERS --nKnowX 5 --prior_betax flat --save_root ${save_root}-bayes
# python ./main.py --tasks summarize --summarize_save_fn summ.xlsx --save_root ${save_root}-bayes


##################################################################################################################################################
#
# NEW: test
#
##################################################################################################################################################
# OTHERS="--prevalence  --OR 1.25 1.5 1.75 2.0 2.25 2.5 2.75 3.0 --nrepeat 1000 --ncores 20 --nSamples 100 --n_knowX_balance"

# test without Z
# python ./simulate_data.py --nSamples 100 --n_knowX_balance --nrepeat 100 \
#     --nKnowX 10 --prevalence 0.5 --OR 3.0 --save_prefix prev05_OR3_X10_sigx1_sige1
# python ./run_simulation.py --target_data "./data/prev05_OR3_X10_sigx1_sige1*.h5" --ncores 20 --solver blackjax \
#     --save_prefix noinfo_prior --prior_betax_std inf  # best
# python ./run_simulation.py --target_data ./data/prev05_OR3_X10_sigx1_sige1*.h5 --ncores 20 --solver blackjax \
#     --save_prefix weak_prior --prior_betax_std 100


# test with Z
# python ./simulate_data.py --nSamples 100 --n_knowX_balance --nrepeat 100 \
#     --nKnowX 10 --prevalence 0.5 --OR 3.0 --betaz 1.0 1.5 2.0  --save_prefix prev05_OR3_X10_sigx1_sige1_Z3
# python ./run_simulation.py --target_data "./data/prev05_OR3_X10_sigx1_sige1_Z3*.h5" --ncores 20 --solver blackjax --ntunes 2000 --ndraws 2000 \
#     --save_prefix noinfo_prior --prior_betax_std inf --prior_betaz_std inf
# python ./run_simulation.py --target_data "./data/prev05_OR3_X10_sigx1_sige1_Z3*.h5" --ncores 20 --solver blackjax --ntunes 2000 --ndraws 2000 \
#     --save_prefix weak_prior --prior_betax_std 100 --prior_betaz_std 100

# test continue
# python ./simulate_data.py --nSamples 100 --nrepeat 100 \
#     --nKnowX 10 --betax 1.0 --type_outcome continue --save_prefix conti_prev05_OR3_X10_sigx1_sige1
# python ./run_simulation.py --target_data "./data/conti_prev05_OR3_X10_sigx1_sige1*.h5" --ncores 20 --solver blackjax --ntunes 2000 --ndraws 2000 \
#     --save_prefix multi_imp_weak_prior --prior_betax_std 10  --type_outcome continue --multi_imp
# python ./run_simulation.py --target_data "./data/conti_prev05_OR3_X10_sigx1_sige1*.h5" --ncores 20 --solver blackjax --ntunes 2000 --ndraws 2000 \
#     --save_prefix weak_prior --prior_betax_std 10  --type_outcome continue


# test survival
# python ./simulate_data.py --nSamples 100 --nrepeat 100 \
#     --nKnowX 10 --betax 1.0 --type_outcome survival --save_prefix surv_2_prev05_OR3_X10_sigx1_sige1
# python ./run_simulation.py --target_data "./data/surv_2_prev05_OR3_X10_sigx1_sige1_2024-09-24*.h5" --ncores 20 --solver blackjax --ntunes 2000 --ndraws 2000 \
#     --save_prefix weak_prior --prior_betax_std 10  --type_outcome survival # best

##################################################################################################################################################
#
# NEW: formal experiments
#
##################################################################################################################################################

# binary outcome, without Z, normal scenario
# for p in 0.25 0.5; do
#     for or in 1.25 1.5 1.75 2.0 2.25 2.5 2.75 3.0; do
#         save_prefix="binary-woZ-Prev$(echo $p |tr '.' ',')-OR$(echo $or |tr '.' ',')"
#         rm ~/.pytensor -rf
#         python ./simulate_data.py --type_outcome binary --nSamples 100 --n_knowX_balance --nrepeat 100 --nKnowX 10 --prevalence $p --OR $or --save_prefix $save_prefix
#         python ./run_simulation.py --type_outcome binary --target_data "./data/${save_prefix}*.h5" --ncores 20 --solver blackjax --ntunes 2000 --ndraws 2000 --save_prefix weak_info_wo_multi_imp
#     done
# done
# python ./summary_simulation.py --res_files "weak_info_wo_multi_imp_binary-woZ*.nc" --index_name_configs OR --column_name_configs Prev


# continue outcome, without Z, normal scenario
# for beta0 in 0.5 1.0; do
#     for betax in 0. 0.5 1.0 1.5 2.0 2.5 3.0 3.5; do
#         save_prefix="continue-woZ-beta0$(echo $beta0 |tr '.' ',')-betax$(echo $betax |tr '.' ',')"
#         rm ~/.pytensor -rf
#         python ./simulate_data.py --type_outcome continue --nSamples 100 --nrepeat 100 --nKnowX 10 --betax $betax --beta0 $beta0 --save_prefix $save_prefix
#         python ./run_simulation.py --type_outcome continue --target_data "./data/${save_prefix}*.h5" --ncores 20 --solver blackjax --ntunes 2000 --ndraws 2000 --save_prefix weak_info_wo_multi_imp
#     done
# done
# python ./summary_simulation.py --res_files "weak_info_wo_multi_imp_continue-woZ*.nc" --index_name_configs betax --column_name_configs beta0

# survival outcome, without Z, normal scenario
for beta0 in 0.5 1.0; do
    for betax in 0. 0.5 1.0 1.5 2.0 2.5 3.0 3.5; do
        save_prefix="survival-woZ-beta0$(echo $beta0 |tr '.' ',')-betax$(echo $betax |tr '.' ',')"
        rm ~/.pytensor -rf
        python ./simulate_data.py --type_outcome survival --nSamples 100 --nrepeat 100 --nKnowX 10 --betax $betax --beta0 $beta0 --save_prefix $save_prefix
        python ./run_simulation.py --type_outcome survival --target_data "./data/${save_prefix}*.h5" --ncores 20 --solver blackjax --ntunes 2000 --ndraws 2000 --save_prefix weak_info_wo_multi_imp
    done
done
python ./summary_simulation.py --res_files "weak_info_wo_multi_imp_survival-woZ*.nc" --index_name_configs betax --column_name_configs beta0