
def create_submit_file(env_name):

    
    submit_file = '+Group   = "GRAD"\n'
    submit_file += '+Project = "AI_ROBOTICS"\n'
    submit_file += '+ProjectDescription = "Learning from suboptimal demonstrations"\n'
    submit_file += '+GPUJob = true\n'
    submit_file += 'universe = vanilla\n'
    submit_file += 'Requirements=(TARGET.GTX1080==true)\n'
    submit_file += 'request_GPUs = 1\n'
    submit_file += 'notify_user = dsbrown\n'
    submit_file += 'Notification = Complete\n'
    submit_file += 'getenv = true\n'
    submit_file += 'Executable = run_' + env_name + '_rl\n'
    submit_file += 'Arguments = $(Process)\n'
    submit_file += 'Error = /scratch/cluster/dsbrown/CondorOutput/' + env_name + '_$(Process)_err\n'
    submit_file += 'Output = /scratch/cluster/dsbrown/CondorOutput/' + env_name + '_$(Process)_out\n'
    submit_file += 'Log = /scratch/cluster/dsbrown/CondorOutput/' + env_name + '_$(Process)_condor.log\n'   
    submit_file += 'Queue 5\n'
    print(submit_file)
    f = open(env_name + "_jobsubmit",'w')
    f.write(submit_file)
    f.close()
    
    
 
#envs = ['mspacman', 'videopinball', 'hero', 'beamrider', 'qbert', 'seaquest', 'breakout', 'spaceinvaders', 'pong', 'enduro' ]
envs = ['mspacman', 'videopinball', 'hero', 'beamrider', 'qbert', 'breakout', 'spaceinvaders', 'pong', 'enduro' ]
#envs = ['mspacman']
for e in envs:
    print("+"*20)
    print(e)
    print("+"*20)
    create_submit_file(e)
