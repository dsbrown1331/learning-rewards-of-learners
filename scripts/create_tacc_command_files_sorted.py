def create_command_file(envname):
    command_str = "module load tacc-singularity\n"
    command_str += 'singularity exec --nv ${SINGULARITY_CACHEDIR}/tacc-maverick-ml-latest.simg python -c "import torch; print(torch.cuda.is_available())"\n'
    command_str += "singularity exec --nv ${SINGULARITY_CACHEDIR}/tacc-maverick-ml-latest.simg python LearnAtariNoviceSnippetsSorted.py --env_name=" + envname +" --seed=0 --reward_model_path=$WORK/output/learned_rewards/" + envname + "_12_sorted_pref.params"
    print(command_str)
    f = open("commands_" + envname,'w')
    f.write(command_str)
    f.close()
    
envs = ['mspacman', 'videopinball', 'hero', 'beamrider', 'qbert', 'seaquest', 'breakout', 'spaceinvaders', 'pong', 'enduro' ]
for e in envs:
    print("+"*20)
    print(e)
    print("+"*20)
    create_command_file(e)
