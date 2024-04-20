# continue test, without z, one
# python main.py --name test --nrepeat 1000 --ncore 20 --log error

# continue, without z
# python main.py --name continue_wo_z --nrepeat 1000 --ncore 20 --log error \
#     -nsps 100 150 200 250 -rxps 0.1 0.15 0.2 -bx 0.0 1.0 2.0
# binary, without z
python main.py --name binary_wo_z --outcome_type binary --nrepeat 100 --ncore 10 --log error --gem \
    -nsps 100 150 200 250 -rxps 0.1 0.15 0.2 -bx 0.0 1.0 2.0 -pr 0.5