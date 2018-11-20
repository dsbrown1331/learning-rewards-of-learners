import matplotlib
matplotlib.use('Agg')
from imgcat import imgcat
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import gym
from pathlib import Path
from tqdm import tqdm
from siamese_ranker import Model
from siamese_ranker import PPO2Agent

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

def get_traj(env,agent):
    obs, actions, rewards = [env.reset()], [], [0.]
    while True:
        action = agent.act(obs[-1], rewards[-1], False)
        ob, reward, done, _ = env.step(action)

        obs.append(ob)
        actions.append(action)
        rewards.append(reward)

        if done: break

    return np.stack(obs,axis=0), np.array(actions), np.array(rewards)


def calc_reward(ranker,traj_a,traj_b,batch_size=16,steps=20):
    def random_crop_traj(traj):
        obs,actions,rewards = traj

        if len(obs)-steps <= 0: # Trajectories are too short.
            _ = np.tile(obs[-1:],(steps-len(obs),1))
            return np.concatenate([obs,_],axis=0)

        t = np.random.randint(len(obs)-steps)
        return obs[t:t+steps]

    b_x, b_y = [], []
    for _ in range(batch_size):
        x = random_crop_traj(traj_a)
        y = random_crop_traj(traj_b)

        b_x.append(x.reshape(-1))
        b_y.append(y.reshape(-1))

    # if x(current traj) is better than y(past traj), than sigmoid(logits) will be close to 1.
    sess = tf.get_default_session()
    logits = sess.run(ranker.logits,
                      feed_dict={
                          ranker.x : np.array(b_x).astype(np.float32),
                          ranker.y : np.array(b_y).astype(np.float32)
                      })
    #vals = 1. / (1. + np.exp(-1. * logits))
    #vals = (vals - 0.5) * 2.
    #reward = np.mean(vals)

    reward = np.mean(logits)
    return reward

if __name__ == "__main__":
    env_name = 'Walker2d-v2'
    env = gym.make(env_name)

    sess = tf.InteractiveSession()

    ranker = Model(env.observation_space.shape[0],128)

    saver = tf.train.Saver(var_list=ranker.parameters(train=False),max_to_keep=0)
    saver.restore(sess,'./log/%s/last.ckpt'%(env_name))

    agents = [
        RandomAgent(env.action_space),
    ]
    p = Path('./learner/models/%s/checkpoints/'%(env_name.split('-')[0].lower()))
    for i,path in enumerate(sorted(p.glob('?????'))):
        agents.append(PPO2Agent(env,'mujoco',str(path),gpu=False))
    assert len(agents)>1

    reward_a = []
    reward_b = []
    ranking_vals = []

    for _ in tqdm(range(1000)):
        traj_a = get_traj(env,np.random.choice(agents))
        traj_b = get_traj(env,np.random.choice(agents))

        reward_a.append(np.sum(traj_a[2]))
        reward_b.append(np.sum(traj_b[2]))
        ranking_vals.append(calc_reward(ranker,traj_a,traj_b))

    reward_diff = np.array(reward_a) - np.array(reward_b)
    print(reward_diff)
    print(ranking_vals)

    fig,ax = plt.subplots()
    ax.plot(reward_diff,ranking_vals,'o')
    imgcat(fig)
    plt.savefig('%s_siamese_ranker_raw_logits.png'%(env_name.split('-')[0].lower()))
    plt.close(fig)
