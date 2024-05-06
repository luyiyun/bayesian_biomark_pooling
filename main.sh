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
python main.py --name binary_wo_z_IS --outcome_type binary --nrepeat 200 --ncore 2 --log error --gem \
    -bs is --delta2 0.01 --gpu -pr 0.5 --root ./results/binary_IS_wo_z_nr200 \
    -nsps 100 -rxps 0.1 0.15 0.2 -bx 0.0 0.5 1.0