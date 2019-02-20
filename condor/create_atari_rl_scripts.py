def create_script_file(env_name):
    if env_name == "spaceinvaders":
        env_id = "SpaceInvadersNoFrameskip-v4"
    elif env_name == "mspacman":
        env_id = "MsPacmanNoFrameskip-v4"
    elif env_name == "videopinball":
        env_id = "VideoPinballNoFrameskip-v4"
    elif env_name == "beamrider":
        env_id = "BeamRiderNoFrameskip-v4"
    else:
        env_id = env_name[0].upper() + env_name[1:] + "NoFrameskip-v4"
    
    script = '#!/usr/bin/env bash\n'
    script += 'source ~/.bashrc\n'
    script += 'conda activate deeplearning\n'
    script += "OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=/scratch/cluster/dsbrown/tflogs/" + env_name + "20env_$1 python -m baselines.run --alg=ppo2 --env=" + env_id + " --custom_reward pytorch --custom_reward_path /u/dsbrown/Code/learning-rewards-of-learners/learner/learned_models/"
    if env_name == "seaquest":
        script += env_name + "_12_5_sorted_pref.params"
    else:
        script += env_name + "_12_sorted_pref.params"
    script +=  " --seed $1 --num_timesteps=4e7 --save_interval=200 --num_env 20"
    
    print(script)
    f = open("run_" + env_name + "_rl",'w')
    f.write(script)
    f.close()
    

#envs = ['mspacman', 'videopinball', 'hero', 'beamrider', 'qbert', 'seaquest', 'breakout', 'spaceinvaders', 'pong', 'enduro' ]
envs = ['hero', 'beamrider', 'qbert', 'breakout', 'spaceinvaders', 'pong', 'enduro', 'seaquest' ]

for e in envs:
    print("+"*20)
    print(e)
    print("+"*20)
    create_script_file(e)
