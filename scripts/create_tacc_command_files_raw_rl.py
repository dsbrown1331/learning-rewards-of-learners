def create_command_file(env_name):
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
    command_str = "module load tacc-singularity\n"
    command_str += "SINGULARITYENV_OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' SINGULARITYENV_OPENAI_LOGDIR=$WORK/tflogs/"+ env_name + "_raw-ppo-2 singularity exec --nv ${SINGULARITY_CACHEDIR}/tacc-maverick-ml-latest.simg python -m baselines.run --alg=ppo2 --env=" + env_id + " --custom_reward pytorch --custom_reward_path $WORK/output/learned_rewards/" + env_name + "_12_raw_pref.params --num_timesteps=5e7 --save_interval=200\n"
    print(command_str)
    f = open("commands_RL_raw_" + env_name,'w')
    f.write(command_str)
    f.close()
    
envs = ['mspacman', 'videopinball', 'hero', 'beamrider', 'qbert', 'seaquest', 'breakout', 'spaceinvaders', 'pong', 'enduro' ]
for e in envs:
    print("+"*20)
    print(e)
    print("+"*20)
    create_command_file(e)
