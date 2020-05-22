# section 1 question 2 runs
python cs285/scripts/run_hw1_behavior_cloning.py -epf cs285/policies/experts/Ant.pkl -env Ant-v2 -exp s1q2 -ed cs285/expert_data/expert_data_Ant-v2.pkl --eval_batch_size 10000
python cs285/scripts/run_hw1_behavior_cloning.py -epf cs285/policies/experts/HalfCheetah.pkl -env HalfCheetah-v2 -exp s1q2 -ed cs285/expert_data/expert_data_HalfCheetah-v2.pkl --eval_batch_size 10000
python cs285/scripts/run_hw1_behavior_cloning.py -epf cs285/policies/experts/Hopper.pkl -env Hopper-v2 -exp s1q2 -ed cs285/expert_data/expert_data_Hopper-v2.pkl --eval_batch_size 10000
python cs285/scripts/run_hw1_behavior_cloning.py -epf cs285/policies/experts/Humanoid.pkl -env Humanoid-v2 -exp s1q2 -ed cs285/expert_data/expert_data_Humanoid-v2.pkl --eval_batch_size 10000
python cs285/scripts/run_hw1_behavior_cloning.py -epf cs285/policies/experts/Walker2d.pkl -env Walker2d-v2 -exp s1q2 -ed cs285/expert_data/expert_data_Walker2d-v2.pkl --eval_batch_size 10000

# section 1 question 3 runs (WARNING: This can take several minutes)
python cs285/scripts/run_hw1_behavior_cloning.py -epf cs285/policies/experts/Hopper.pkl -env Hopper-v2 -exp s1q3 -ed cs285/expert_data/expert_data_Hopper-v2.pkl --eval_batch_size 10000 --num_agent_train_steps_per_iter 100
python cs285/scripts/run_hw1_behavior_cloning.py -epf cs285/policies/experts/Hopper.pkl -env Hopper-v2 -exp s1q3 -ed cs285/expert_data/expert_data_Hopper-v2.pkl --eval_batch_size 10000 --num_agent_train_steps_per_iter 1000
python cs285/scripts/run_hw1_behavior_cloning.py -epf cs285/policies/experts/Hopper.pkl -env Hopper-v2 -exp s1q3 -ed cs285/expert_data/expert_data_Hopper-v2.pkl --eval_batch_size 10000 --num_agent_train_steps_per_iter 10000
python cs285/scripts/run_hw1_behavior_cloning.py -epf cs285/policies/experts/Hopper.pkl -env Hopper-v2 -exp s1q3 -ed cs285/expert_data/expert_data_Hopper-v2.pkl --eval_batch_size 10000 --num_agent_train_steps_per_iter 100000
python cs285/scripts/run_hw1_behavior_cloning.py -epf cs285/policies/experts/Hopper.pkl -env Hopper-v2 -exp s1q3 -ed cs285/expert_data/expert_data_Hopper-v2.pkl --eval_batch_size 10000 --num_agent_train_steps_per_iter 1000000

# section 2 runs
python cs285/scripts/run_hw1_behavior_cloning.py -epf cs285/policies/experts/Hopper.pkl -env Hopper-v2 -exp s2q2 -ed cs285/expert_data/expert_data_Hopper-v2.pkl --eval_batch_size 10000
python cs285/scripts/run_hw1_behavior_cloning.py -epf cs285/policies/experts/Hopper.pkl -env Hopper-v2 -exp s2q2 -n 10 --do_dagger -ed cs285/expert_data/expert_data_Hopper-v2.pkl --eval_batch_size 10000 --video_log_freq -1 --seed 1
python cs285/scripts/run_hw1_behavior_cloning.py -epf cs285/policies/experts/Hopper.pkl -env Hopper-v2 -exp s2q2 -n 10 --do_dagger -ed cs285/expert_data/expert_data_Hopper-v2.pkl --eval_batch_size 10000 --video_log_freq -1 --seed 2
python cs285/scripts/run_hw1_behavior_cloning.py -epf cs285/policies/experts/Hopper.pkl -env Hopper-v2 -exp s2q2 -n 10 --do_dagger -ed cs285/expert_data/expert_data_Hopper-v2.pkl --eval_batch_size 10000 --video_log_freq -1 --seed 3

# plot figures
pythonw cs285/scripts/plot.py