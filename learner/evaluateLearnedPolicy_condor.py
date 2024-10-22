import os
import sys
import pickle
import gym
import time
import numpy as np
import random
import torch
from run_test import *
#import matplotlib.pylab as plt
import argparse

def evaluate_learned_policy(env_name, tflogdir, checkpoint, rep):
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

    env_type = "atari"

    stochastic = True

    #env id, env type, num envs, and seed
    env = make_vec_env(env_id, 'atari', 1, 0,
                       wrapper_kwargs={
                           'clip_rewards':False,
                           'episode_life':False,
                       })



    env = VecFrameStack(env, 4)


    agent = PPO2Agent(env, env_type, stochastic)  #defaults to stochastic = False (deterministic policy)
    #agent = RandomAgent(env.action_space)

    learning_returns = []
    model_path = "/scratch/cluster/dsbrown/tflogs/" + env_name + tflogdir + str(rep) + "/checkpoints/" + checkpoint
    #model_path = "/home/dsbrown/Code/learning-rewards-of-learners/learner/models/spaceinvaders/checkpoints/" + checkpoint
    print(model_path)

    agent.load(model_path)
    episode_count = 10
    for i in range(episode_count):
        done = False
        traj = []
        r = 0

        ob = env.reset()
        #traj.append(ob)
        #print(ob.shape)
        steps = 0
        acc_reward = 0
        while True:
            action = agent.act(ob, r, done)
            #print(action)
            ob, r, done, _ = env.step(action)

            #print(ob.shape)
            steps += 1
            #print(steps)
            acc_reward += r[0]
            if done:
                print("steps: {}, return: {}".format(steps,acc_reward))
                break
        learning_returns.append(acc_reward)



    env.close()
    #tf.reset_default_graph()



    return learning_returns

    #return([np.max(learning_returns), np.min(learning_returns), np.mean(learning_returns)])

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--seed', default=1234, help="random seed for experiments")
    parser.add_argument('--env_name', default='', help='Select the environment name to run, i.e. pong')
    parser.add_argument('--tflogdir', default='20env_', help='postfix to append to env_name to get tflogs')
    parser.add_argument('--checkpoint', default='', help='checkpoint to run eval on')
    parser.add_argument('--rep', default='', help='which trial to evaluate')
    args = parser.parse_args()
    env_name = args.env_name
    tflogdir = args.tflogdir
    #set seeds
    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    checkpoint = args.checkpoint
    rep = args.rep
    print("*"*10)
    print(env_name)
    print("*"*10)
    returns = evaluate_learned_policy(env_name, tflogdir, checkpoint, rep)
    #write returns to file
    f = open("./eval/" + env_name + tflogdir + checkpoint + "_" + rep + "eval.txt",'w')
    for r in returns:
        f.write("{}\n".format(r))
    f.close()
