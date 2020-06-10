# run experiments (only one random seed for now)
python cs285/scripts/run_hw2_policy_gradient.py --env_name CartPole-v0 -n 100 -b 1000 -dsa -l 1 -s 32 --exp_name sb_no_rtg_dsa
python cs285/scripts/run_hw2_policy_gradient.py --env_name CartPole-v0 -n 100 -b 1000 -rtg -dsa -l 1 -s 32 --exp_name sb_rtg_dsa
python cs285/scripts/run_hw2_policy_gradient.py --env_name CartPole-v0 -n 100 -b 1000 -rtg -l 1 -s 32 --exp_name sb_rtg_na
python cs285/scripts/run_hw2_policy_gradient.py --env_name CartPole-v0 -n 100 -b 5000 -dsa -l 1 -s 32 --exp_name lb_no_rtg_dsa
python cs285/scripts/run_hw2_policy_gradient.py --env_name CartPole-v0 -n 100 -b 5000 -rtg -dsa -l 1 -s 32 --exp_name lb_rtg_dsa
python cs285/scripts/run_hw2_policy_gradient.py --env_name CartPole-v0 -n 100 -b 5000 -rtg -l 1 -s 32 --exp_name lb_rtg_na

# plot
pythonw cs285/scripts/plot.py -pb 3 -ws 3