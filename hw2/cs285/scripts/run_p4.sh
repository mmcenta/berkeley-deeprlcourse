python cs285/scripts/run_hw2_policy_gradient.py --env_name  InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 100 -lr 0.01 -rtg --exp_name ip_b100_r1e-2
python cs285/scripts/run_hw2_policy_gradient.py --env_name  InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 100 -lr 0.02 -rtg --exp_name ip_b100_r2e-2
python cs285/scripts/run_hw2_policy_gradient.py --env_name  InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 100 -lr 0.03 -rtg --exp_name ip_b100_r3e-2
python cs285/scripts/run_hw2_policy_gradient.py --env_name  InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 100 -lr 0.04 -rtg --exp_name ip_b100_r4e-2
python cs285/scripts/run_hw2_policy_gradient.py --env_name  InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 100 -lr 0.05 -rtg --exp_name ip_b100_r5e-2
pythonw cs285/scripts/plot.py -pb 4 -ws 3