import argparse
import sys

import gym
from gym import wrappers, logger

sys.path.append('./baselines/')
import baselines.ppo2.ppo2 as ppo2
from baselines.common.policies import build_policy
from baselines.common.cmd_util import make_vec_env
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

class PPO2Agent(object):
    def __init__(self, env, path):
        ob_space = env.observation_space
        ac_space = env.action_space

        policy = build_policy(env,'cnn')
        make_model = lambda : ppo2.Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=1, nbatch_train=1,
                        nsteps=1, ent_coef=0., vf_coef=0.,
                        max_grad_norm=0.)
        self.model = make_model()
        if path:
            self.model.load(path)

    def act(self, observation, reward, done):
        a,v,state,neglogp = self.model.step(observation)
        return a

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_id', default='', help='Select the environment to run')
    parser.add_argument('--model_path', default='')
    parser.add_argument('--episode_count', default=100)
    parser.add_argument('--record_video', action='store_true')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    #env = gym.make(args.env_id)

    #env id, env type, num envs, and seed
    env = make_vec_env(args.env_id, 'atari', 1, 0,
                       wrapper_kwargs={
                           'clip_rewards':False,
                           'episode_life':False,
                       })
    if args.record_video:
        env = VecVideoRecorder(env,'.',lambda steps: True, 10000) # Always record every episode
    env = VecFrameStack(env, 4)

    agent = PPO2Agent(env,args.model_path)
    #agent = RandomAgent(env.action_space)

    episode_count = args.episode_count
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        steps = 0
        acc_reward = 0
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            steps += 1
            acc_reward += reward
            if done:
                print(steps,acc_reward)
                break

    env.close()
    env.env.close()
