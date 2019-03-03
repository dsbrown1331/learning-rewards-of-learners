
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
    submit_file += 'Executable = timepref_trex_' + env_name + '\n'
    submit_file += 'Error = /scratch/cluster/dsbrown/CondorOutput/' + env_name + '_learn_reward_timepref_err\n'
    submit_file += 'Output = /scratch/cluster/dsbrown/CondorOutput/' + env_name + '_learn_reward_timepref_out\n'
    submit_file += 'Log = /scratch/cluster/dsbrown/CondorOutput/' + env_name + '_   learn_reward_timepref_condor.log\n'   
    submit_file += 'Queue\n'
    print(submit_file)
    f = open(env_name + "_trex_timepref_jobsubmit",'w')
    f.write(submit_file)
    f.close()
    
    
 
envs = ['beamrider', 'breakout', 'enduro', 'hero', 'pong', 'qbert', 'seaquest', 'spaceinvaders']
for e in envs:
    print("+"*20)
    print(e)
    print("+"*20)
    create_submit_file(e)
