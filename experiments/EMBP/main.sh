set -e # 一旦出现错误，立即停止运行，并打印出错误信息。

# nrepeat=1000
# ncore=20
runtool="uv run"
# runtool="python"

# >>> ===================== test =======================
# 基础配置
# nrepeat=10
# ncore=1
# runtool="uv run"

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
nrepeat=10
ncore=1
data_dir=./example_binary_2/data
ana_dir=./example_binary_2/results
eval_fn=eval_results.csv
outcome_type=binary
# $runtool main.py simulate -ot $outcome_type -od $data_dir --seed 1 \
#     --n_samples 100 --ratio_observed_x 0.1 -pr 0.5 --OR 2.0 -nr $nrepeat
$runtool main.py analyze -ot $outcome_type -dd $data_dir -od $ana_dir -nc $ncore --binary_solve vi
$runtool main.py evaluate -ad $ana_dir -of $eval_fn

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
# seed=4000
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

# ================ scenario10: continue outcome, with Z ================
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


# ================ scenario1003: continue binary, without Z ================
# seed=0
# num_samples=(100 150 200 250)
# ratio_observed_x=(0.1)
# ORs=(1.25 1.5 1.75 2 2.25 2.5 2.75 3)
# for n in ${num_samples[@]}; do
#     for rx in ${ratio_observed_x[@]}; do
#         for OR in ${ORs[@]}; do
#             seed=$((seed+1))
#             echo "<==========> n=$n, rx=$rx, OR=$OR, seed=$seed"
#             data_dir=./scenario1003/data_binary_wo_z_${n}_${rx}_${OR}
#             ana_dir=./scenario1003/ana_binary_wo_z_${n}_${rx}_${OR}
#             eval_fn=eval_results.csv
#             $runtool main.py simulate -ot binary -od $data_dir --seed $seed \
#                 --n_samples $n --ratio_observed_x $rx --OR $OR -nr $nrepeat -pr 0.25
#             $runtool main.py analyze -ot binary -dd $data_dir -od $ana_dir -nc $ncore --methods xonly naive
#             $runtool main.py evaluate -ad $ana_dir -of $eval_fn
#         done
#     done
# done
# $runtool main.py summarize -efp "./scenario1003/ana_binary_wo_z_*/eval_results.csv" -of ./scenario1003/summary1.xlsx \
#     -sp  n_sample_per_studies OR

# ================ scenario1004: continue binary, without Z ================
# seed=1000
# num_samples=(100 150 200 250)
# ratio_observed_x=(0.1)
# ORs=(1.25 1.5 1.75 2 2.25 2.5 2.75 3)
# for n in ${num_samples[@]}; do
#     for rx in ${ratio_observed_x[@]}; do
#         for OR in ${ORs[@]}; do
#             seed=$((seed+1))
#             echo "<==========> n=$n, rx=$rx, OR=$OR, seed=$seed"
#             data_dir=./scenario1004/data_binary_wo_z_${n}_${rx}_${OR}
#             ana_dir=./scenario1004/ana_binary_wo_z_${n}_${rx}_${OR}
#             eval_fn=eval_results.csv
#             python main.py simulate -ot binary -od $data_dir --seed $seed \
#                 --n_samples $n --ratio_observed_x $rx --OR $OR --sigma2_x 10 --sigma2_e 10 -nr $nrepeat -pr 025
#             python main.py analyze -ot binary -dd $data_dir -od $ana_dir -nc $ncore --methods xonly naive
#             python main.py evaluate -ad $ana_dir -of $eval_fn
#         done
#     done
# done
# $runtool main.py summarize -efp "./scenario1004/ana_binary_wo_z_*/eval_results.csv" -of ./scenario1004/summary1.xlsx \
#     -sp  n_sample_per_studies OR

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