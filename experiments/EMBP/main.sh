set -e # 一旦出现错误，立即停止运行，并打印出错误信息。

nrepeat=1000
ncore=20
# runtool="uv run"
runtool="python"

# >>> ===================== test =======================
# 基础配置
# nrepeat=10
# ncore=1
# runtool="uv run"

# 测试方差问题
# data_dir=./test/data_continue_wo_z_muti5
# ana_dir=./test/ana_continue_wo_z_muti5
# python main.py simulate -ot continue -od $data_dir --seed 0 \
#     --n_samples 100 --ratio_observed_x 0.2 --beta_x 1 --sigma2_x 5 --sigma2_e 5 -nr 1000 

# ================ scenario5: continue outcome, without Z ================
# seed=0
# num_samples=(100 150 200 250)
# ratio_observed_x=(0.2)
# beta_x=(0.0 0.5 1.0 1.5 2.0 2.5 3.0)
# for n in ${num_samples[@]}; do
#     for rx in ${ratio_observed_x[@]}; do
#         for bx in ${beta_x[@]}; do
#             seed=$((seed+1))
#             echo "<==========> n=$n, rx=$rx, bx=$bx, seed=$seed"
#             data_dir=./test/scenario5/data_continue_wo_z_${n}_${rx}_${bx}
#             ana_dir=./test/scenario5/ana_continue_wo_z_${n}_${rx}_${bx}
#             eval_fn=eval_results.csv
#             $runtool main.py simulate -ot continue -od $data_dir --seed $seed \
#                 --n_samples $n --ratio_observed_x $rx --beta_x $bx -nr $nrepeat
#             # $runtool main.py analyze -ot continue -dd $data_dir -od $ana_dir -nc $ncore
#             # $runtool main.py evaluate -ad $ana_dir -of $eval_fn
#         done
#     done
# done

# ================ scenario9: continue outcome, without Z ================
# seed=0
# num_samples=(100 150 200 250)
# ratio_observed_x=(0.2)
# beta_x=(0.0 0.5 1.0 1.5 2.0 2.5 3.0)
# for n in ${num_samples[@]}; do
#     for rx in ${ratio_observed_x[@]}; do
#         for bx in ${beta_x[@]}; do
#             seed=$((seed+1))
#             echo "<==========> n=$n, rx=$rx, bx=$bx, seed=$seed"
#             data_dir=./test/scenario9/data_continue_wo_z_${n}_${rx}_${bx}
#             ana_dir=./test/scenario9/ana_continue_wo_z_${n}_${rx}_${bx}
#             eval_fn=eval_results.csv
#             python main.py simulate -ot continue -od $data_dir --seed $seed \
#                 --n_samples $n --ratio_observed_x $rx --beta_x $bx --sigma2_x 5 --sigma2_e 5 -nr $nrepeat
#         done
#     done
# done



# 单次实验测试
# data_dir=./example_1/data
# ana_dir=./example_1/results
# eval_fn=eval_results.csv
# outcome_type=continue
# $runtool main.py simulate -ot $outcome_type -od $data_dir --seed 1 \
#     --n_samples 100 --ratio_observed_x 0.1 --beta_x 1 -nr $nrepeat
# $runtool main.py analyze -ot $outcome_type -dd $data_dir -od $ana_dir -nc $ncore
# $runtool main.py evaluate -ad $ana_dir -of $eval_fn
# data_dir=./example_2/data
# ana_dir=./example_2/results
# eval_fn=eval_results.csv
# outcome_type=continue
# $runtool main.py simulate -ot $outcome_type -od $data_dir --seed 2 \
#     --n_samples 100 --ratio_observed_x 0.1 --beta_x 2 -nr $nrepeat
# $runtool main.py analyze -ot $outcome_type -dd $data_dir -od $ana_dir -nc $ncore
# $runtool main.py evaluate -ad $ana_dir -of $eval_fn

# 单次实验测试 (binary outcome)
# nrepeat=10
# ncore=1
# data_dir=./example_binary_2/data
# ana_dir=./example_binary_2/results
# eval_fn=eval_results.csv
# outcome_type=binary
# # $runtool main.py simulate -ot $outcome_type -od $data_dir --seed 1 \
# #     --n_samples 100 --ratio_observed_x 0.1 -pr 0.5 --OR 2.0 -nr $nrepeat
# $runtool main.py analyze -ot $outcome_type -dd $data_dir -od $ana_dir -nc $ncore --binary_solve vi -epb
# $runtool main.py evaluate -ad $ana_dir -of $eval_fn
# $runtool main.py analyze -ot $outcome_type -dd $data_dir -od $ana_dir -nc $ncore --binary_solve vi
# $runtool main.py evaluate -ad $ana_dir -of $eval_fn

# 单次实验测试
# data_dir=./example_continue_xonly_1/data
# ana_dir=./example_continue_xonly_1/results
# eval_fn=eval_results.csv
# outcome_type=continue
# for n_samples in 100 300 500 700 1000; do
#     data_dir_i=${data_dir}/nsamples_${n_samples}
#     ana_dir_i=${ana_dir}/nsamples_${n_samples}
#     $runtool main.py simulate -ot $outcome_type -od $data_dir_i --seed 2 \
#         --n_samples $n_samples --ratio_observed_x 0.1 --beta_x 1 -nr 1000
#     $runtool main.py analyze -ot $outcome_type -dd $data_dir_i -od $ana_dir_i -nc 10 --methods xonly
#     $runtool main.py evaluate -ad $ana_dir_i -of $eval_fn
# done

# 单次实验测试 (with Z)
# data_dir=./example_with_z/data
# ana_dir=./example_with_z/results
# eval_fn=eval_results.csv
# outcome_type=continue
# $runtool main.py simulate -ot $outcome_type -od $data_dir --seed 1 \
#     --n_samples 100 --ratio_observed_x 0.2 --beta_x 1 --beta_z 1 2 -nr 10
# $runtool main.py analyze -ot $outcome_type -dd $data_dir -od $ana_dir -nc 1 --methods embp
# $runtool main.py evaluate -ad $ana_dir -of $eval_fn

# 循环运行多种实验配置
# seed=0
# num_samples=(100 150 200 250)
# ratio_observed_x=(0.1 0.15 0.2)
# beta_x=(0.0 1.0 2.0)
# for n in ${num_samples[@]}; do
#     for rx in ${ratio_observed_x[@]}; do
#         for bx in ${beta_x[@]}; do
#             seed=$((seed+1))
#             echo "<==========> n=$n, rx=$rx, bx=$bx, seed=$seed"
#             data_dir=./example/data_continue_wo_z_${n}_${rx}_${bx}
#             ana_dir=./example/ana_continue_wo_z_${n}_${rx}_${bx}
#             eval_fn=eval_results.csv
#             $runtool main.py simulate -ot continue -od $data_dir --seed $seed \
#                 --n_samples $n --ratio_observed_x $rx --beta_x $bx -nr $nrepeat
#             $runtool main.py analyze -ot continue -dd $data_dir -od $ana_dir -nc $ncore
#             $runtool main.py evaluate -ad $ana_dir -of $eval_fn
#         done
#     done
# done

# 总结上面循环得到的所有结果，并生成一个excel文件。
# 上面运行的结果需要使用通配符来匹配（-efp, --evaluated_file_pattern）。
# $runtool main.py summarize -efp "./example/ana_continue_wo_z_*/eval_results.csv" -of ./example/summary.xlsx \
#     -sp ratio_observed_x beta_x n_samples
# data_dir=./example/data
# ana_dir=./example/results
# eval_fn=./example/eval_results.csv
# outcome_type=continue
# $runtool main.py simulate -ot $outcome_type -od $data_dir --n_samples 100 --ratio_observed_x 0.1 --beta_x 1 -nr 100
# $runtool main.py analyze -ot $outcome_type -dd $data_dir -od $ana_dir -nc 10
# $runtool main.py evaluate -ad $ana_dir -of $eval_fn

# <<< ===================== test =======================

# ================ scenario5: continue outcome, without Z ================
# seed=0
# num_samples=(100 150 200 250)
# ratio_observed_x=(0.2)
# beta_x=(0.0 0.5 1.0 1.5 2.0 2.5 3.0)
# for n in ${num_samples[@]}; do
#     for rx in ${ratio_observed_x[@]}; do
#         for bx in ${beta_x[@]}; do
#             seed=$((seed+1))
#             echo "<==========> n=$n, rx=$rx, bx=$bx, seed=$seed"
#             data_dir=./scenario5/data_continue_wo_z_${n}_${rx}_${bx}
#             ana_dir=./scenario5/ana_continue_wo_z_${n}_${rx}_${bx}
#             eval_fn=eval_results.csv
#             $runtool main.py simulate -ot continue -od $data_dir --seed $seed \
#                 --n_samples $n --ratio_observed_x $rx --beta_x $bx -nr $nrepeat
#             $runtool main.py analyze -ot continue -dd $data_dir -od $ana_dir -nc $ncore
#             $runtool main.py evaluate -ad $ana_dir -of $eval_fn
#         done
#     done
# done
# $runtool main.py summarize -efp "./scenario5/ana_continue_wo_z_*/eval_results.csv" -of ./scenario5/summary.xlsx \
#     -sp  n_sample_per_studies betax

# ================ scenario6: continue outcome, without Z ================
# seed=1000
# num_samples=(100 150 200 250)
# ratio_observed_x=(0.2)
# beta_x=(0.0 0.5 1.0 1.5 2.0 2.5 3.0)
# for n in ${num_samples[@]}; do
#     for rx in ${ratio_observed_x[@]}; do
#         for bx in ${beta_x[@]}; do
#             seed=$((seed+1))
#             echo "<==========> n=$n, rx=$rx, bx=$bx, seed=$seed"
#             data_dir=./scenario6/data_continue_wo_z_${n}_${rx}_${bx}
#             ana_dir=./scenario6/ana_continue_wo_z_${n}_${rx}_${bx}
#             eval_fn=eval_results.csv
#             python main.py simulate -ot continue -od $data_dir --seed $seed \
#                 --n_samples $n --ratio_observed_x $rx --beta_x $bx --sigma2_x 10 --sigma2_e 10 -nr $nrepeat
#             python main.py analyze -ot continue -dd $data_dir -od $ana_dir -nc $ncore
#             python main.py evaluate -ad $ana_dir -of $eval_fn
#         done
#     done
# done
# $runtool main.py summarize -efp "./scenario6/ana_continue_wo_z_*/eval_results.csv" -of ./scenario6/summary.xlsx \
#     -sp  n_sample_per_studies betax

# ================ scenario7: continue outcome, without Z ================
# seed=2000
# num_samples=(200 250)
# ratio_observed_x=(0.1)
# beta_x=(0.0 0.5 1.0 1.5 2.0 2.5 3.0)
# for n in ${num_samples[@]}; do
#     for rx in ${ratio_observed_x[@]}; do
#         for bx in ${beta_x[@]}; do
#             seed=$((seed+1))
#             echo "<==========> n=$n, rx=$rx, bx=$bx, seed=$seed"
#             data_dir=./scenario7/data_continue_wo_z_${n}_${rx}_${bx}
#             ana_dir=./scenario7/ana_continue_wo_z_${n}_${rx}_${bx}
#             eval_fn=eval_results.csv
#             python main.py simulate -ot continue -od $data_dir --seed $seed \
#                 --n_samples $n --ratio_observed_x $rx --beta_x $bx -nr $nrepeat
#             python main.py analyze -ot continue -dd $data_dir -od $ana_dir -nc $ncore
#             python main.py evaluate -ad $ana_dir -of $eval_fn
#         done
#     done
# done
# $runtool main.py summarize -efp "./scenario7/ana_continue_wo_z_*/eval_results.csv" -of ./scenario7/summary.xlsx \
#     -sp  n_sample_per_studies betax

# ================ scenario8: continue outcome, without Z ================
# seed=3000
# num_samples=(100 150 200 250)
# # num_samples=(100 150)
# ratio_observed_x=(3300)
# beta_x=(0.0 0.5 1.0 1.5 2.0 2.5 3.0)
# for n in ${num_samples[@]}; do
#     for rx in ${ratio_observed_x[@]}; do
#         for bx in ${beta_x[@]}; do
#             seed=$((seed+1))
#             echo "<==========> n=$n, rx=$rx, bx=$bx, seed=$seed"
#             data_dir=./scenario8/data_continue_wo_z_${n}_${rx}_${bx}
#             ana_dir=./scenario8/ana_continue_wo_z_${n}_${rx}_${bx}
#             eval_fn=eval_results.csv
#             python main.py simulate -ot continue -od $data_dir --seed $seed \
#                 --n_samples $n --ratio_observed_x 0.3 0.3 0 0 --beta_x $bx -nr $nrepeat
#             python main.py analyze -ot continue -dd $data_dir -od $ana_dir -nc $ncore
#             python main.py evaluate -ad $ana_dir -of $eval_fn
#         done
#     done
# done
# $runtool main.py summarize -efp "./scenario8/ana_continue_wo_z_*/eval_results.csv" -of ./scenario8/summary.xlsx \
#     -sp  n_sample_per_studies betax

# ================ scenario9: continue outcome, without Z ================
# seed=0
# num_samples=(100 150 200 250)
# ratio_observed_x=(0.2)
# beta_x=(0.0 0.5 1.0 1.5 2.0 2.5 3.0)
# for n in ${num_samples[@]}; do
#     for rx in ${ratio_observed_x[@]}; do
#         for bx in ${beta_x[@]}; do
#             seed=$((seed+1))
#             echo "<==========> n=$n, rx=$rx, bx=$bx, seed=$seed"
#             data_dir=./scenario9/data_continue_wo_z_${n}_${rx}_${bx}
#             ana_dir=./scenario9/ana_continue_wo_z_${n}_${rx}_${bx}
#             eval_fn=eval_results.csv
#             python main.py simulate -ot continue -od $data_dir --seed $seed \
#                 --n_samples $n --ratio_observed_x $rx --beta_x $bx --sigma2_x 5 --sigma2_e 5 -nr $nrepeat
#             python main.py analyze -ot continue -dd $data_dir -od $ana_dir -nc $ncore
#             python main.py evaluate -ad $ana_dir -of $eval_fn
#         done
#     done
# done
# $runtool main.py summarize -efp "./scenario9/ana_continue_wo_z_*/eval_results.csv" -of ./scenario9/summary.xlsx \
#     -sp  n_sample_per_studies betax

# ================ scenario10: continue outcome, with Z-1 ================
# seed=5000
# num_samples=(100 150 200 250)
# ratio_observed_x=(0.2)
# beta_x=(0.0 0.5 1.0 1.5 2.0 2.5 3.0)
# for n in ${num_samples[@]}; do
#     for rx in ${ratio_observed_x[@]}; do
#         for bx in ${beta_x[@]}; do
#             seed=$((seed+1))
#             echo "<==========> n=$n, rx=$rx, bx=$bx, seed=$seed"
#             data_dir=./scenario10/data_continue_wo_z_${n}_${rx}_${bx}
#             ana_dir=./scenario10/ana_continue_wo_z_${n}_${rx}_${bx}
#             eval_fn=eval_results.csv
#             $runtool main.py simulate -ot continue -od $data_dir --seed $seed \
#                 --n_samples $n --ratio_observed_x $rx --beta_x $bx --beta_z 1 -nr $nrepeat
#             $runtool main.py analyze -ot continue -dd $data_dir -od $ana_dir -nc $ncore
#             $runtool main.py evaluate -ad $ana_dir -of $eval_fn
#         done
#     done
# done
# $runtool main.py summarize -efp "./scenario10/ana_continue_wo_z_*/eval_results.csv" -of ./scenario10/summary.xlsx \
#     -sp  n_sample_per_studies betax

# # ================ scenario11: continue outcome, with Z-2 ================
# seed=5000
# num_samples=(100 150 200 250)
# ratio_observed_x=(0.2)
# beta_x=(0.0 0.5 1.0 1.5 2.0 2.5 3.0)
# for n in ${num_samples[@]}; do
#     for rx in ${ratio_observed_x[@]}; do
#         for bx in ${beta_x[@]}; do
#             seed=$((seed+1))
#             echo "<==========> n=$n, rx=$rx, bx=$bx, seed=$seed"
#             data_dir=./scenario11/data_continue_wo_z_${n}_${rx}_${bx}
#             ana_dir=./scenario11/ana_continue_wo_z_${n}_${rx}_${bx}
#             eval_fn=eval_results.csv
#             $runtool main.py simulate -ot continue -od $data_dir --seed $seed \
#                 --n_samples $n --ratio_observed_x $rx --beta_x $bx --beta_z 1 1 -nr $nrepeat
#             $runtool main.py analyze -ot continue -dd $data_dir -od $ana_dir -nc $ncore
#             $runtool main.py evaluate -ad $ana_dir -of $eval_fn
#         done
#     done
# done
# $runtool main.py summarize -efp "./scenario11/ana_continue_wo_z_*/eval_results.csv" -of ./scenario11/summary.xlsx \
#     -sp  n_sample_per_studies betax

# # ================ scenario12: continue outcome, with Z-3 ================
# seed=5000
# num_samples=(100 150 200 250)
# ratio_observed_x=(0.2)
# beta_x=(0.0 0.5 1.0 1.5 2.0 2.5 3.0)
# for n in ${num_samples[@]}; do
#     for rx in ${ratio_observed_x[@]}; do
#         for bx in ${beta_x[@]}; do
#             seed=$((seed+1))
#             echo "<==========> n=$n, rx=$rx, bx=$bx, seed=$seed"
#             data_dir=./scenario12/data_continue_wo_z_${n}_${rx}_${bx}
#             ana_dir=./scenario12/ana_continue_wo_z_${n}_${rx}_${bx}
#             eval_fn=eval_results.csv
#             $runtool main.py simulate -ot continue -od $data_dir --seed $seed \
#                 --n_samples $n --ratio_observed_x $rx --beta_x $bx --beta_z 1 1 1 -nr $nrepeat
#             $runtool main.py analyze -ot continue -dd $data_dir -od $ana_dir -nc $ncore
#             $runtool main.py evaluate -ad $ana_dir -of $eval_fn
#         done
#     done
# done
# $runtool main.py summarize -efp "./scenario12/ana_continue_wo_z_*/eval_results.csv" -of ./scenario12/summary.xlsx \
#     -sp  n_sample_per_studies betax

    
# # ================ scenario13: continue outcome, with Z-4 ================
# seed=5000
# num_samples=(100 150 200 250)
# ratio_observed_x=(0.2)
# beta_x=(0.0 0.5 1.0 1.5 2.0 2.5 3.0)
# for n in ${num_samples[@]}; do
#     for rx in ${ratio_observed_x[@]}; do
#         for bx in ${beta_x[@]}; do
#             seed=$((seed+1))
#             echo "<==========> n=$n, rx=$rx, bx=$bx, seed=$seed"
#             data_dir=./scenario13/data_continue_wo_z_${n}_${rx}_${bx}
#             ana_dir=./scenario13/ana_continue_wo_z_${n}_${rx}_${bx}
#             eval_fn=eval_results.csv
#             $runtool main.py simulate -ot continue -od $data_dir --seed $seed \
#                 --n_samples $n --ratio_observed_x $rx --beta_x $bx --beta_z 1 1 1 1 -nr $nrepeat
#             $runtool main.py analyze -ot continue -dd $data_dir -od $ana_dir -nc $ncore
#             $runtool main.py evaluate -ad $ana_dir -of $eval_fn
#         done
#     done
# done
# $runtool main.py summarize -efp "./scenario13/ana_continue_wo_z_*/eval_results.csv" -of ./scenario13/summary.xlsx \
#     -sp  n_sample_per_studies betax

# # ================ scenario14: continue outcome, with Z-5 ================
# seed=5000
# num_samples=(100 150 200 250)
# ratio_observed_x=(0.2)
# beta_x=(0.0 0.5 1.0 1.5 2.0 2.5 3.0)
# for n in ${num_samples[@]}; do
#     for rx in ${ratio_observed_x[@]}; do
#         for bx in ${beta_x[@]}; do
#             seed=$((seed+1))
#             echo "<==========> n=$n, rx=$rx, bx=$bx, seed=$seed"
#             data_dir=./scenario14/data_continue_wo_z_${n}_${rx}_${bx}
#             ana_dir=./scenario14/ana_continue_wo_z_${n}_${rx}_${bx}
#             eval_fn=eval_results.csv
#             $runtool main.py simulate -ot continue -od $data_dir --seed $seed \
#                 --n_samples $n --ratio_observed_x $rx --beta_x $bx --beta_z 1 1 1 1 1 -nr $nrepeat
#             $runtool main.py analyze -ot continue -dd $data_dir -od $ana_dir -nc $ncore
#             $runtool main.py evaluate -ad $ana_dir -of $eval_fn
#         done
#     done
# done
# $runtool main.py summarize -efp "./scenario14/ana_continue_wo_z_*/eval_results.csv" -of ./scenario14/summary.xlsx \
#     -sp  n_sample_per_studies betax

# # ================ scenario2001: continue outcome, without Z ================
# seed=0
# num_samples=(100 150 200 250)
# ratio_observed_x=(0.1 0.2)
# beta_x=(0.0 0.5 1.0 1.5 2.0 2.5 3.0)
# for n in ${num_samples[@]}; do
#     for rx in ${ratio_observed_x[@]}; do
#         for bx in ${beta_x[@]}; do
#             seed=$((seed+1))
#             echo "<==========> n=$n, rx=$rx, bx=$bx, seed=$seed"
#             data_dir=./scenario15/data_continue_wo_z_${n}_${rx}_${bx}
#             ana_dir=./scenario15/ana_continue_wo_z_${n}_${rx}_${bx}
#             eval_fn=eval_results.csv
#             $runtool main.py simulate -ot continue -od $data_dir --seed $seed \
#                 --n_samples $n --ratio_observed_x $rx --beta_x $bx -nr $nrepeat --sigma2_y 5
#             $runtool main.py analyze -ot continue -dd $data_dir -od $ana_dir -nc $ncore
#             $runtool main.py evaluate -ad $ana_dir -of $eval_fn
#         done
#     done
# done
# $runtool main.py summarize -efp "./scenario15/ana_continue_wo_z_*/eval_results.csv" -of ./scenario15/summary.xlsx \
#     -sp  betax n_samples ratio_observed_x  

# ================ scenario2002: continue outcome, without Z ================
# seed=0
# nrepeat=1000
# num_samples=(100 150 200 250)
# rx=0.2
# beta_x=(0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0)
# sigma2_xs=(1 10)
# for n in ${num_samples[@]}; do
#     for bx in ${beta_x[@]}; do
#         for sigma2_x in ${sigma2_xs[@]}; do
#             seed=$((seed+1))
#             sigma2_y=$(echo "9 * ($bx ^ 2) * $sigma2_x" | bc -l)
#             echo "<==========> n=$n, rx=$rx, bx=$bx, sigma2_x=$sigma2_x, sigma2_y=$sigma2_y, seed=$seed"
#             data_dir=./scenario2002/data_continue_wo_z_${n}_${sigma2_x}_${bx}
#             $runtool main.py simulate -ot continue -od $data_dir --seed $seed \
#                 --n_samples $n --ratio_observed_x $rx --beta_x $bx -nr $nrepeat --sigma2_x $sigma2_x --sigma2_e $sigma2_x --sigma2_y $sigma2_y
#         done
#     done
# done

# num_samples=(100 200)
# rx=0.2
# beta_x=(0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0)
# sigma2_xs=(1 10)
# for n in ${num_samples[@]}; do
#     for bx in ${beta_x[@]}; do
#         for sigma2_x in ${sigma2_xs[@]}; do
#             seed=$((seed+1))
#             sigma2_y=$(echo "9 * ($bx ^ 2) * $sigma2_x" | bc -l)
#             echo "<==========> n=$n, rx=$rx, bx=$bx, sigma2_x=$sigma2_x, sigma2_y=$sigma2_y"
#             data_dir=./scenario2002/data_continue_wo_z_${n}_${sigma2_x}_${bx}
#             ana_dir=./scenario2002/ana_continue_wo_z_${n}_${sigma2_x}_${bx}
#             eval_fn=eval_results.csv
#             $runtool main.py analyze -ot continue -dd $data_dir -od $ana_dir -nc $ncore
#             $runtool main.py evaluate -ad $ana_dir -of $eval_fn
#         done
#     done
# done
# $runtool main.py summarize -efp "./scenario2002/ana_continue_wo_z_*/eval_results.csv" -of ./scenario2002/summary.xlsx \
#     -sp  betax n_samples sigma2_x 

# ================ scenario2003: continue outcome, without Z ================
# seed=0
# num_samples=(100 150 200 250)
# rx=0.2
# beta_x=(0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0)
# sigma2_xs=(1 10)
# for n in ${num_samples[@]}; do
#     for bx in ${beta_x[@]}; do
#         for sigma2_x in ${sigma2_xs[@]}; do
#             seed=$((seed+1))
#             sigma2_y=$(echo "19 * ($bx ^ 2) * $sigma2_x" | bc -l)
#             echo "<==========> n=$n, rx=$rx, bx=$bx, sigma2_x=$sigma2_x, sigma2_y=$sigma2_y, seed=$seed"
#             data_dir=./scenario2003/data_continue_wo_z_${n}_${sigma2_x}_${bx}
#             $runtool main.py simulate -ot continue -od $data_dir --seed $seed \
#                 --n_samples $n --ratio_observed_x $rx --beta_x $bx -nr $nrepeat --sigma2_x $sigma2_x --sigma2_e $sigma2_x --sigma2_y $sigma2_y
#         done
#     done
# done

# num_samples=(100 200)
# rx=0.2
# beta_x=(0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0)
# sigma2_xs=(1 10)
# for n in ${num_samples[@]}; do
#     for bx in ${beta_x[@]}; do
#         for sigma2_x in ${sigma2_xs[@]}; do
#             seed=$((seed+1))
#             sigma2_y=$(echo "19 * ($bx ^ 2) * $sigma2_x" | bc -l)
#             echo "<==========> n=$n, rx=$rx, bx=$bx, sigma2_x=$sigma2_x, sigma2_y=$sigma2_y"
#             data_dir=./scenario2003/data_continue_wo_z_${n}_${sigma2_x}_${bx}
#             ana_dir=./scenario2003/ana_continue_wo_z_${n}_${sigma2_x}_${bx}
#             eval_fn=eval_results.csv
#             $runtool main.py analyze -ot continue -dd $data_dir -od $ana_dir -nc $ncore
#             $runtool main.py evaluate -ad $ana_dir -of $eval_fn
#         done
#     done
# done
# $runtool main.py summarize -efp "./scenario2003/ana_continue_wo_z_*/eval_results.csv" -of ./scenario2003/summary.xlsx \
#     -sp  betax n_samples sigma2_x

# ================ scenario2004: continue outcome, without Z ================
# seed=0
# num_samples=(100 150 200 250)
# rx=2200
# beta_x=(0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0)
# sigma2_xs=(1)
# for n in ${num_samples[@]}; do
#     for bx in ${beta_x[@]}; do
#         for sigma2_x in ${sigma2_xs[@]}; do
#             seed=$((seed+1))
#             sigma2_y=$(echo "9 * ($bx ^ 2) * $sigma2_x" | bc -l)
#             echo "<==========> n=$n, rx=$rx, bx=$bx, sigma2_x=$sigma2_x, sigma2_y=$sigma2_y, seed=$seed"
#             data_dir=./scenario2004/data_continue_wo_z_${n}_${sigma2_x}_${bx}
#             ana_dir=./scenario2004/ana_continue_wo_z_${n}_${sigma2_x}_${bx}
#             eval_fn=eval_results.csv
#             $runtool main.py simulate -ot continue -od $data_dir --seed $seed \
#                 --n_samples $n --ratio_observed_x 0.2 0.2 0 0 --beta_x $bx -nr $nrepeat --sigma2_x $sigma2_x --sigma2_e $sigma2_x --sigma2_y $sigma2_y
#             $runtool main.py analyze -ot continue -dd $data_dir -od $ana_dir -nc $ncore
#             $runtool main.py evaluate -ad $ana_dir -of $eval_fn
#         done
#     done
# done
# $runtool main.py summarize -efp "./scenario2004/ana_continue_wo_z_*/eval_results.csv" -of ./scenario2004/summary.xlsx \
#     -sp  betax n_samples sigma2_x

# ================ scenario2005: continue outcome, without Z ================
# seed=0
# num_samples=(100 200)
# rx=4400
# beta_x=(0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0)
# sigma2_xs=(1)
# for n in ${num_samples[@]}; do
#     for bx in ${beta_x[@]}; do
#         for sigma2_x in ${sigma2_xs[@]}; do
#             seed=$((seed+1))
#             sigma2_y=$(echo "19 * ($bx ^ 2) * $sigma2_x" | bc -l)
#             echo "<==========> n=$n, rx=$rx, bx=$bx, sigma2_x=$sigma2_x, sigma2_y=$sigma2_y, seed=$seed"
#             data_dir=./scenario2005/data_continue_wo_z_${n}_${sigma2_x}_${bx}
#             ana_dir=./scenario2005/ana_continue_wo_z_${n}_${sigma2_x}_${bx}
#             eval_fn=eval_results.csv
#             $runtool main.py simulate -ot continue -od $data_dir --seed $seed \
#                 --n_samples $n --ratio_observed_x 0.4 0.4 0 0 --beta_x $bx -nr $nrepeat --sigma2_x $sigma2_x --sigma2_e $sigma2_x --sigma2_y $sigma2_y
#             $runtool main.py analyze -ot continue -dd $data_dir -od $ana_dir -nc $ncore
#             $runtool main.py evaluate -ad $ana_dir -of $eval_fn
#         done
#     done
# done
# $runtool main.py summarize -efp "./scenario2005/ana_continue_wo_z_*/eval_results.csv" -of ./scenario2005/summary.xlsx \
#     -sp  betax n_samples

# ================ scenario2006: continue outcome, without Z ================
# seed=0
# num_samples=(100 200)
# rx=0.4
# beta_x=(0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0)
# sigma2_xs=(1)
# for n in ${num_samples[@]}; do
#     for bx in ${beta_x[@]}; do
#         for sigma2_x in ${sigma2_xs[@]}; do
#             seed=$((seed+1))
#             sigma2_y=$(echo "9 * ($bx ^ 2) * $sigma2_x" | bc -l)
#             echo "<==========> n=$n, rx=$rx, bx=$bx, sigma2_x=$sigma2_x, sigma2_y=$sigma2_y, seed=$seed"
#             data_dir=./scenario2006/data_continue_wo_z_${n}_${sigma2_x}_${bx}
#             ana_dir=./scenario2006/ana_continue_wo_z_${n}_${sigma2_x}_${bx}
#             eval_fn=eval_results.csv
#             $runtool main.py simulate -ot continue -od $data_dir --seed $seed \
#                 --n_samples $n --ratio_observed_x $rx --beta_x $bx -nr $nrepeat --sigma2_x $sigma2_x --sigma2_e $sigma2_x --sigma2_y $sigma2_y
#             $runtool main.py analyze -ot continue -dd $data_dir -od $ana_dir -nc $ncore
#             $runtool main.py evaluate -ad $ana_dir -of $eval_fn
#         done
#     done
# done
# $runtool main.py summarize -efp "./scenario2006/ana_continue_wo_z_*/eval_results.csv" -of ./scenario2006/summary.xlsx \
#     -sp  betax n_samples

# ================ scenario2007: continue outcome, without Z ================
# seed=0
# num_samples=(100 200)
# rx=0.3
# beta_x=(0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0)
# sigma2_xs=(1)
# for n in ${num_samples[@]}; do
#     for bx in ${beta_x[@]}; do
#         for sigma2_x in ${sigma2_xs[@]}; do
#             seed=$((seed+1))
#             sigma2_y=$(echo "9 * ($bx ^ 2) * $sigma2_x" | bc -l)
#             echo "<==========> n=$n, rx=$rx, bx=$bx, sigma2_x=$sigma2_x, sigma2_y=$sigma2_y, seed=$seed"
#             data_dir=./scenario2007/data_continue_wo_z_${n}_${sigma2_x}_${bx}
#             ana_dir=./scenario2007/ana_continue_wo_z_${n}_${sigma2_x}_${bx}
#             eval_fn=eval_results.csv
#             $runtool main.py simulate -ot continue -od $data_dir --seed $seed \
#                 --n_samples $n --ratio_observed_x $rx --beta_x $bx -nr $nrepeat --sigma2_x $sigma2_x --sigma2_e $sigma2_x --sigma2_y $sigma2_y
#             $runtool main.py analyze -ot continue -dd $data_dir -od $ana_dir -nc $ncore
#             $runtool main.py evaluate -ad $ana_dir -of $eval_fn
#         done
#     done
# done
# $runtool main.py summarize -efp "./scenario2007/ana_continue_wo_z_*/eval_results.csv" -of ./scenario2007/summary.xlsx \
#     -sp  n_samples betax 

# ================ scenario2008: continue outcome, without Z ================
# seed=1000
# num_samples=(100 200)
# rx=0.2
# beta_x=(0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0)
# sigma2_xs=(1)
# for n in ${num_samples[@]}; do
#     for bx in ${beta_x[@]}; do
#         for sigma2_x in ${sigma2_xs[@]}; do
#             seed=$((seed+1))
#             sigma2_y=$(echo "9 * ($bx ^ 2) * $sigma2_x" | bc -l)
#             echo "<==========> n=$n, rx=$rx, bx=$bx, sigma2_x=$sigma2_x, sigma2_y=$sigma2_y, seed=$seed"
#             data_dir=./scenario2008/data_continue_wo_z_${n}_${sigma2_x}_${bx}
#             ana_dir=./scenario2008/ana_continue_wo_z_${n}_${sigma2_x}_${bx}
#             eval_fn=eval_results.csv
#             $runtool main.py simulate -ot continue -od $data_dir --seed $seed \
#                 --n_samples $n --ratio_observed_x $rx --beta_x $bx -nr $nrepeat --sigma2_x $sigma2_x --sigma2_e 10 --sigma2_y $sigma2_y
#             $runtool main.py analyze -ot continue -dd $data_dir -od $ana_dir -nc $ncore
#             $runtool main.py evaluate -ad $ana_dir -of $eval_fn
#         done
#     done
# done
# $runtool main.py summarize -efp "./scenario2008/ana_continue_wo_z_*/eval_results.csv" -of ./scenario2008/summary.xlsx \
#     -sp n_samples  betax 

# # ================ scenario2009: continue outcome, without Z ================
# seed=2000
# num_samples=(100 200)
# rx=0.2
# beta_x=(0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0)
# sigma2_xs=(1)
# for n in ${num_samples[@]}; do
#     for bx in ${beta_x[@]}; do
#         for sigma2_x in ${sigma2_xs[@]}; do
#             seed=$((seed+1))
#             sigma2_y=$(echo "9 * ($bx ^ 2) * $sigma2_x" | bc -l)
#             echo "<==========> n=$n, rx=$rx, bx=$bx, sigma2_x=$sigma2_x, sigma2_y=$sigma2_y, seed=$seed"
#             data_dir=./scenario2009/data_continue_wo_z_${n}_${sigma2_x}_${bx}
#             ana_dir=./scenario2009/ana_continue_wo_z_${n}_${sigma2_x}_${bx}
#             eval_fn=eval_results.csv
#             $runtool main.py simulate -ot continue -od $data_dir --seed $seed \
#                 --n_samples $n --ratio_observed_x $rx --beta_x $bx -nr $nrepeat --sigma2_x $sigma2_x --sigma2_e 0.1 --sigma2_y $sigma2_y
#             $runtool main.py analyze -ot continue -dd $data_dir -od $ana_dir -nc $ncore
#             $runtool main.py evaluate -ad $ana_dir -of $eval_fn
#         done
#     done
# done
# $runtool main.py summarize -efp "./scenario2009/ana_continue_wo_z_*/eval_results.csv" -of ./scenario2009/summary.xlsx \
#     -sp n_samples  betax 

# ================ scenario2010: continue outcome, without Z ================
# seed=0
# num_samples=(100 200)
# rx=0.1
# beta_x=(0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0)
# sigma2_xs=(1)
# for n in ${num_samples[@]}; do
#     for bx in ${beta_x[@]}; do
#         for sigma2_x in ${sigma2_xs[@]}; do
#             seed=$((seed+1))
#             sigma2_y=$(echo "9 * ($bx ^ 2) * $sigma2_x" | bc -l)
#             echo "<==========> n=$n, rx=$rx, bx=$bx, sigma2_x=$sigma2_x, sigma2_y=$sigma2_y, seed=$seed"
#             data_dir=./scenario2010/data_continue_wo_z_${n}_${sigma2_x}_${bx}
#             ana_dir=./scenario2010/ana_continue_wo_z_${n}_${sigma2_x}_${bx}
#             eval_fn=eval_results.csv
#             $runtool main.py simulate -ot continue -od $data_dir --seed $seed \
#                 --n_samples $n --ratio_observed_x $rx --beta_x $bx -nr $nrepeat --sigma2_x $sigma2_x --sigma2_e $sigma2_x --sigma2_y $sigma2_y
#             $runtool main.py analyze -ot continue -dd $data_dir -od $ana_dir -nc $ncore
#             $runtool main.py evaluate -ad $ana_dir -of $eval_fn
#         done
#     done
# done
# $runtool main.py summarize -efp "./scenario2010/ana_continue_wo_z_*/eval_results.csv" -of ./scenario2010/summary.xlsx \
#     -sp  n_samples betax 
# n=100
# pr=0.05
# rx=0.1
# OR=1.25
# data_dir=./test/data_binary_wo_z_${n}_${pr}_${rx}_${OR}
# ana_dir=./test/ana_binary_wo_z_${n}_${pr}_${rx}_${OR}
# $runtool main.py simulate -ot binary -od $data_dir --seed 0 --n_samples $n --ratio_observed_x $rx --OR $OR -nr 10 -pr $pr
# $runtool main.py analyze -ot binary -dd $data_dir -od $ana_dir -nc 1
# eval_fn=eval_results.csv
# ana_dir=./test/ana_binary_wo_z_${n}_${pr}_${rx}_${OR}
# $runtool main.py evaluate -ad $ana_dir -of $eval_fn

# 单个测试
# data_dir=./test/data_binary_wo_z_${n}_${pr}_${rx}_${OR}
# ana_dir_vi=./test/ana_binary_wo_z_vi_${n}_${pr}_${rx}_${OR}
# $runtool main.py simulate -ot binary -od $data_dir --seed $seed --n_samples $n --ratio_observed_x $rx --OR $OR -nr 10 -pr $pr
# $runtool main.py analyze -ot binary  -dd $data_dir -od $ana_dir_vi -nc $ncore --binary_solve vi
# $runtool main.py evaluate -ad $ana_dir_vi -of $eval_fn

# # ================ scenario1001: continue binary, without Z ================
# seed=0
# ncore=20
# nrepeat=100
# num_samples=(100)
# prevalences=(0.1 0.3 0.5)
# ratio_observed_x=(0.1 0.2)
# ORs=(1.25 1.5 1.75 2 2.25 2.5 2.75 3)
# for n in ${num_samples[@]}; do
#     for pr in ${prevalences[@]};do
#         for rx in ${ratio_observed_x[@]}; do
#             for OR in ${ORs[@]}; do
#                 seed=$((seed+1))
#                 echo "<==========> n=$n, pr=$pr, rx=$rx, OR=$OR, seed=$seed"
#                 data_dir=./scenario1001/data_binary_wo_z_${n}_${pr}_${rx}_${OR}
#                 ana_dir=./scenario1001/ana_binary_wo_z_${n}_${pr}_${rx}_${OR}
#                 eval_fn=eval_results.csv
#                 $runtool main.py simulate -ot binary -od $data_dir --seed $seed --n_samples $n --ratio_observed_x $rx --OR $OR -nr $nrepeat -pr $pr
#                 $runtool main.py analyze -ot binary  -dd $data_dir -od $ana_dir -nc $ncore --binary_solve vi
#                 $runtool main.py evaluate -ad $ana_dir -of $eval_fn
#             done
#         done
#     done
# done

# $runtool main.py summarize -efp "./scenario1001/ana_binary_wo_z_*/eval_results.csv" -of ./scenario1001/summary.xlsx \
#     -sp  prevalence n_knowX_per_studies OR

# # ================ scenario1002: continue binary, without Z ================
# seed=1000
# ncore=20
# num_samples=(200)
# prevalences=(0.1 0.3 0.5)
# ratio_observed_x=(0.1 0.2)
# ORs=(1.25 1.5 1.75 2 2.25 2.5 2.75 3)
# for n in ${num_samples[@]}; do
#     for pr in ${prevalences[@]};do
#         for rx in ${ratio_observed_x[@]}; do
#             for OR in ${ORs[@]}; do
#                 seed=$((seed+1))
#                 echo "<==========> n=$n, pr=$pr, rx=$rx, OR=$OR, seed=$seed"
#                 data_dir=./scenario1002/data_binary_wo_z_${n}_${pr}_${rx}_${OR}
#                 ana_dir=./scenario1002/ana_binary_wo_z_${n}_${pr}_${rx}_${OR}
#                 eval_fn=eval_results.csv
#                 $runtool main.py simulate -ot binary -od $data_dir --seed $seed --n_samples $n --ratio_observed_x $rx --OR $OR -nr $nrepeat -pr $pr
#                 $runtool main.py analyze -ot binary  -dd $data_dir -od $ana_dir -nc $ncore --binary_solve vi
#                 $runtool main.py evaluate -ad $ana_dir -of $eval_fn
#             done
#         done
#     done
# done

# $runtool main.py summarize -efp "./scenario1002/ana_binary_wo_z_*/eval_results.csv" -of ./scenario1002/summary.xlsx \
#     -sp  prevalence n_knowX_per_studies OR

# # ================ scenario1003: continue binary, without Z ================
# seed=2000
# ncore=20
# nrepeat=100
# num_samples=(100)
# prevalences=(0.1 0.3 0.5)
# ratio_observed_x=(0.1 0.2)
# ORs=(1.25 1.5 1.75 2 2.25 2.5 2.75 3)
# for n in ${num_samples[@]}; do
#     for pr in ${prevalences[@]};do
#         for rx in ${ratio_observed_x[@]}; do
#             for OR in ${ORs[@]}; do
#                 seed=$((seed+1))
#                 echo "<==========> n=$n, pr=$pr, rx=$rx, OR=$OR, seed=$seed"
#                 data_dir=./scenario1003/data_binary_wo_z_${n}_${pr}_${rx}_${OR}
#                 ana_dir=./scenario1003/ana_binary_wo_z_${n}_${pr}_${rx}_${OR}
#                 eval_fn=eval_results.csv
#                 $runtool main.py simulate -ot binary -od $data_dir --seed $seed --n_samples $n --ratio_observed_x $rx --sigma2_x 10 --sigma2_e 10 --OR $OR -nr $nrepeat -pr $pr
#                 $runtool main.py analyze -ot binary  -dd $data_dir -od $ana_dir -nc $ncore --binary_solve vi
#                 $runtool main.py evaluate -ad $ana_dir -of $eval_fn
#             done
#         done
#     done
# done

# $runtool main.py summarize -efp "./scenario1003/ana_binary_wo_z_*/eval_results.csv" -of ./scenario1003/summary.xlsx \
#     -sp  prevalence n_knowX_per_studies OR

# ================ scenario1004: continue binary, without Z ================
# seed=3000
# ncore=20
# num_samples=(200)
# prevalences=(0.1 0.3 0.5)
# ratio_observed_x=(0.1 0.2)
# ORs=(1.25 1.5 1.75 2 2.25 2.5 2.75 3)
# for n in ${num_samples[@]}; do
#     for pr in ${prevalences[@]};do
#         for rx in ${ratio_observed_x[@]}; do
#             for OR in ${ORs[@]}; do
#                 seed=$((seed+1))
#                 echo "<==========> n=$n, pr=$pr, rx=$rx, OR=$OR, seed=$seed"
#                 data_dir=./scenario1004/data_binary_wo_z_${n}_${pr}_${rx}_${OR}
#                 ana_dir=./scenario1004/ana_binary_wo_z_${n}_${pr}_${rx}_${OR}
#                 eval_fn=eval_results.csv
#                 $runtool main.py simulate -ot binary -od $data_dir --seed $seed --n_samples $n --ratio_observed_x $rx --sigma2_x 10 --sigma2_e 10 --OR $OR -nr $nrepeat -pr $pr
#                 $runtool main.py analyze -ot binary  -dd $data_dir -od $ana_dir -nc $ncore --binary_solve vi
#                 $runtool main.py evaluate -ad $ana_dir -of $eval_fn
#             done
#         done
#     done
# done

# $runtool main.py summarize -efp "./scenario1004/ana_binary_wo_z_*/eval_results.csv" -of ./scenario1004/summary.xlsx \
#     -sp  prevalence n_knowX_per_studies OR


# ================ scenario1002: continue binary, without Z ================
# seed=1000
# num_samples=(100 150 200 250)
# prevalences=(0.25 0.5 0.05)
# ratio_observed_x=(0.1)
# ORs=(1.25 1.5 1.75 2 2.25 2.5 2.75 3)
# for n in ${num_samples[@]}; do
#     for pr in ${prevalences[@]};do
#         for rx in ${ratio_observed_x[@]}; do
#             for OR in ${ORs[@]}; do
#                 seed=$((seed+1))
#                 echo "<==========> n=$n, pr=$pr, rx=$rx, OR=$OR, seed=$seed"
#                 data_dir=./scenario1002/data_binary_wo_z_${n}_${pr}_${rx}_${OR}
#                 ana_dir_is=./scenario1002/ana_binary_wo_z_is_${n}_${pr}_${rx}_${OR}
#                 ana_dir_lap=./scenario1002/ana_binary_wo_z_lap_${n}_${pr}_${rx}_${OR}
#                 eval_fn=eval_results.csv
#                 python main.py simulate -ot binary -od $data_dir --seed $seed \
#                     --n_samples $n --ratio_observed_x $rx --OR $OR --sigma2_x 10 --sigma2_e 10 -nr 10 -pr $pr
#                 $runtool main.py analyze -ot binary -dd $data_dir -od $ana_dir_lap -nc $ncore 
#                 $runtool main.py analyze -ot binary -dd $data_dir -od $ana_dir_is -nc 1 --gpu -bs is
#                 $runtool main.py analyze -ot $outcome_type -dd $data_dir -od $ana_dir -nc $ncore --binary_solve vi
#                 $runtool main.py evaluate -ad $ana_dir_is -of $eval_fn
#                 $runtool main.py evaluate -ad $ana_dir_lap -of $eval_fn
#             done
#         done
#     done
# done
# $runtool main.py summarize -efp "./scenario1002/ana_binary_wo_z_is*/eval_results.csv" -of ./scenario1002/summary_is.xlsx \
#     -sp  prevalence n_sample_per_studies OR
# $runtool main.py summarize -efp "./scenario1002/ana_binary_wo_z_lap*/eval_results.csv" -of ./scenario1002/summary_lap.xlsx \
#     -sp  prevalence n_sample_per_studies OR

# ================ continue outcome, without Z ================
# num_samples=(100 150 200 250)
# ratio_observed_x=(0.1 0.15 0.2)
# beta_x=(0.0 1.0 2.0)
# for n in ${num_samples[@]}; do
#     for rx in ${ratio_observed_x[@]}; do
#         for bx in ${beta_x[@]}; do
#             echo "<==========> n=$n, rx=$rx, bx=$bx"
#             data_dir=./data/continue_wo_z_${n}_${rx}_${bx}
#             ana_dir=./results/continue_wo_z_${n}_${rx}_${bx}
#             eval_fn=$ana_dir/eval_results.csv
#             $runtool main.py simulate -ot continue -od $data_dir --n_samples $n --ratio_observed_x $rx --beta_x $bx -nr $nrepeat
#             $runtool main.py analyze -ot continue -dd $data_dir -od $ana_dir -nc $ncore
#             $runtool main.py evaluate -ad $ana_dir -of $eval_fn
#         done
#     done
# done

# ================ continue outcome, with Z ================
# xxx

# continue test, without z, one
# python main.py --name test --nrepeat 1000 --ncore 20 --log error

# continue, without z
# python main.py --name continue_wo_z --nrepeat 1000 --ncore 20 --log error \
#     -nsps 100 150 200 250 -rxps 0.1 0.15 0.2 -bx 0.0 1.0 2.0

# Lap binary, without z
# python main.py --name test_binary_wo_z --outcome_type binary --nrepeat 1000 --ncore 20 --log error --gem \
#     -nsps 100 150 200 250 -rxps 0.1 0.15 0.2 -bx 0.0 0.5 1.0 -pr 0.5 -qK 100
# Lap binary, without z, qK=500
# python main.py --name test_binary_wo_z --outcome_type binary --nrepeat 1000 --ncore 10 --log error --gem \
#     -nsps 100 150 200 250 -rxps 0.1 0.15 0.2 -bx 0.0 0.5 1.0 -pr 0.5 -qK 500 --skip 100,0.1,0 100,0.1,0.5 100,0.1,1.0

# 修正，之前m step中估计grad_o时，忘了-Yo
# Lap binary, without z
# python main.py --name binary_wo_z --outcome_type binary --nrepeat 1000 --ncore 20 --log error --gem \
#     --root ./results/binary_wo_z_qK100_fix \
#     -nsps 100 150 200 250 -rxps 0.1 0.15 0.2 -bx 0.0 0.5 1.0 -pr 0.5 -qK 100
# lap binary, without z, qK=500
# python main.py --name test_binary_wo_z --outcome_type binary --nrepeat 1000 --ncore 10 --log error --gem \
#     -nsps 100 150 200 250 -rxps 0.1 0.15 0.2 -bx 0.0 0.5 1.0 -pr 0.5 -qK 500 --skip 100,0.1,0 100,0.1,0.5 100,0.1,1.0

# ISbinary, without z,
# 1. in gpu3
# python main.py --name binary_wo_z_IS --outcome_type binary --nrepeat 200 --ncore 4 --log error --gem \
#     -bs is --delta2 0.01 --gpu -pr 0.5 --root ./results/binary_IS_wo_z_nr200 \
#     -nsps 100 -rxps 0.1 0.15 0.2 -bx 0.0 0.5 1.0
# 2. in gpu2
# python main.py --name binary_wo_z_IS --outcome_type binary --nrepeat 200 --ncore 4 --log error --gem \
#     -bs is --delta2 0.01 --gpu -pr 0.5 --root ./results/binary_IS_wo_z_nr200 \
#     -nsps 150 200 250 -rxps 0.1 0.15 0.2 -bx 0.0 0.5 1.0

# continue, without z, just for EMBP-sem
# python main.py --name continue_wo_z --nrepeat 1000 --ncore 20 --log error \
#     -nsps 100 150 200 250 -rxps 0.1 0.15 0.2 -bx 0.0 1.0 2.0 --methods EMBP --ci_method sem\
#     --root ./results/continue_wo_z_EMBP_sem

# Lap binary, without z just for EMBP-lap-sem
# python main.py --name binary_wo_z --outcome_type binary --nrepeat 1000 --ncore 20 --log error --gem \
#     --root ./results/binary_wo_z_qK100_fix_EMBP_sem --methods EMBP --ci_method sem \
#     -nsps 100 150 200 250 -rxps 0.1 0.15 0.2 -bx 0.0 0.5 1.0 -pr 0.5 -qK 100

# ISbinary, without z, just for EMBP-is-sem
# 1. in cpu test
# python main.py --name binary_wo_z_IS --outcome_type binary --nrepeat 200 --ncore 4 --log error --gem \
#     -bs is --delta2 0.01 -pr 0.5 --root ./results/binary_IS_wo_z_nr200_EMBP_sem_1 \
#     --methods EMBP --ci_method sem \
#     -nsps 100 -rxps 0.1 -bx 0.0
# python main.py --name binary_wo_z_IS --outcome_type binary --nrepeat 200 --ncore 4 --log error --gem \
#     -bs is --delta2 0.01 -pr 0.5 --root ./results/binary_IS_wo_z_nr200_EMBP_sem_2 \
#     --methods EMBP --ci_method sem \
#     -nsps 100 150 200 250 -rxps 0.1 0.15 0.2 -bx 0.0 0.5 1.0 --skip 100,0.1,0.0