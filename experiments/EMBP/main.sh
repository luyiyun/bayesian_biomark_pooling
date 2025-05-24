set -e # 一旦出现错误，立即停止运行，并打印出错误信息。

nrepeat=1000
ncore=1
# runtool="uv run"
runtool="python"

# ================ test ================
# 基础配置
# nrepeat=10
# ncore=10
# runtool="uv run"

# 单次实验测试
# data_dir=./example/data
# ana_dir=./example/results
# eval_fn=eval_results.csv
# outcome_type=continue
# $runtool main.py simulate -ot $outcome_type -od $data_dir --n_samples 100 --ratio_observed_x 0.1 --beta_x 1 -nr 100
# $runtool main.py analyze -ot $outcome_type -dd $data_dir -od $ana_dir -nc 10
# $runtool main.py evaluate -ad $ana_dir -of $eval_fn

# 循环运行多种实验配置
# num_samples=(100 150 200 250)
# ratio_observed_x=(0.1 0.15 0.2)
# beta_x=(0.0 1.0 2.0)
# for n in ${num_samples[@]}; do
#     for rx in ${ratio_observed_x[@]}; do
#         for bx in ${beta_x[@]}; do
#             echo "<==========> n=$n, rx=$rx, bx=$bx"
#             data_dir=./example/data_continue_wo_z_${n}_${rx}_${bx}
#             ana_dir=./example/ana_continue_wo_z_${n}_${rx}_${bx}
#             eval_fn=eval_results.csv
#             $runtool main.py simulate -ot continue -od $data_dir --n_samples $n --ratio_observed_x $rx --beta_x $bx -nr $nrepeat
#             $runtool main.py analyze -ot continue -dd $data_dir -od $ana_dir -nc $ncore
#             $runtool main.py evaluate -ad $ana_dir -of $eval_fn
#         done
#     done
# done

# 总结上面循环得到的所有结果，并生成一个excel文件。
# 上面运行的结果需要使用通配符来匹配（-efp, --evaluated_file_pattern）。
# $runtool main.py summarize -efp "./example/ana_continue_wo_z_*/eval_results.csv" -of ./example/summary.xlsx \
#     -sp ratio_observed_x beta_x n_samples

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