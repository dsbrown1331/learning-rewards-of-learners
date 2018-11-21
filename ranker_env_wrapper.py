import os, sys
import numpy as np
import tensorflow as tf
import gym

# TF Log Level configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Only show ERROR log

class RankerEnvGT(object):
    def __init__(self,
                 env_id):
        """
        Make Environment
        """
        self.env = gym.make(env_id)

        """
        Previous & Current Trajectories to evaluate rewards.
        """
        obs = [self.env.reset()]
        actions = []
        rewards = []
        while True:
            action = self.env.action_space.sample()
            ob, reward, done, _ = self.env.step(action)

            obs.append(ob)
            actions.append(action)
            rewards.append(reward)
            if done: break
        self.past_trajs = [
            (np.stack(obs,axis=0),
             np.stack(actions,axis=0),
             np.stack(rewards,axis=0))]

        self.current_traj = []
        self.accumulate = False
        self.current_trajs = []

    def seed(self,seed):
        return self.env.seed(seed)

    def render(self,mode):
        return self.env.render(mode)

    def reset(self):
        ob = self.env.reset()
        self.current_traj = [(ob,None,None)]

        return ob

    def step(self,action):
        ob, true_r, done, info = self.env.step(action)
        self.current_traj.append((ob,action,true_r))

        if done :
            obs, actions, true_rs = zip(*self.current_traj)
            obs = np.stack(obs,axis=0)
            actions = np.stack(actions[1:],axis=0)
            true_rs = np.stack(true_rs[1:],axis=0) # get rid of the first element (None;placeholder)

            traj = (obs,actions,true_rs)
            if self.accumulate:
                self.current_trajs.append(traj)

            reward = self.calc_reward(traj)
        else:
            traj = None
            reward = 0.

        return ob, reward, done, {'true_reward':true_r,'traj':traj}

    def start_accumulate_trajs(self):
        assert self.accumulate == False, 'it was accumulating'
        self.accumulate = True

    def end_accumulate_trajs(self):
        self.past_trajs = self.current_trajs
        self.current_trajs = []
        self.accumulate = False

    def calc_reward(self,traj):
        true_reward = np.sum(traj[2])
        past_true_reward = np.mean([np.sum(rewards) for _,_,rewards in self.past_trajs])

        return true_reward - past_true_reward

class RankerEnvGTBinary(RankerEnvGT):
    def calc_reward(self,traj):
        true_reward = np.sum(traj[2])
        past_true_reward = np.mean([np.sum(rewards) for _,_,rewards in self.past_trajs])

        return 1. if (true_reward - past_true_reward) > 0 else -1.

class RankerEnv(RankerEnvGT):
    def __init__(self,
                 env_id,
                 RankerModel,
                 model_path):
        super().__init__(env_id)

        """
        Build Ranker Model Graph & Load Parameters
        """
        self.graph = tf.Graph()

        config = tf.ConfigProto(
            device_count = {'GPU': 0}) # Run on CPU
        #config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph,config=config)

        with self.graph.as_default():
            with self.sess.as_default():
                self.ranker_model = RankerModel(self.env.observation_space.shape[0])
                saver = tf.train.Saver(var_list=self.ranker_model.parameters(train=False),max_to_keep=0)
                saver.restore(self.sess,model_path)


    def calc_reward(self,traj,batch_size=16,steps=20):
        b_x, b_y = [], []
        for _ in range(batch_size):
            def _pick_traj_and_crop(trajs):
                idx = np.random.randint(len(trajs))
                obs,actions,rewards = trajs[idx]

                if len(obs)-steps <= 0: # Trajectories are too short.
                    _ = np.tile(obs[-1:],(steps-len(obs),1))
                    return np.concatenate([obs,_],axis=0)

                t = np.random.randint(len(obs)-steps)
                return obs[t:t+steps]

            x = _pick_traj_and_crop([traj])
            y = _pick_traj_and_crop(self.past_trajs)

            b_x.append(x.reshape(-1))
            b_y.append(y.reshape(-1))

        # if x(current traj) is better than y(past traj), than sigmoid(logits) will be close to 1.
        logits = self.sess.run(self.ranker_model.logits,
                               feed_dict={
                                   self.ranker_model.x : np.array(b_x).astype(np.float32),
                                   self.ranker_model.y : np.array(b_y).astype(np.float32)
                               })
        vals = 1. / (1. + np.exp(-1. * logits))
        vals = (vals - 0.5) * 2.
        reward = np.mean(vals)

        return reward

class RankerEnvBinary(RankerEnv):
    def calc_reward(self,traj,batch_size=16,steps=20):
        reward = super().calc_reward(traj,batch_size,steps)

        return 1.0 if reward > 0. else -1.

# Test Code
if __name__ == "__main__":
    from siamese_ranker import Model
    from functools import partial

    #env = RankerEnvGT(
    #    'Hopper-v2')
    #env = RankerEnv(
    env = RankerEnvBinary(
        'Hopper-v2',
        partial(Model,embedding_dims=128,steps=20),
        './log/Hopper-v2/last.ckpt'
    )

    from pathlib import Path
    from siamese_ranker import PPO2Agent
    train_agents = []
    valid_agents = []
    p = Path('./learner/models/hopper/checkpoints/')
    for i,path in enumerate(sorted(p.glob('?????'))):
        agent = PPO2Agent(env.env,'mujoco',str(path),gpu=False)
        if i % 2 == 0 :
            train_agents.append(agent)
        else:
            valid_agents.append(agent)

    np.random.shuffle(train_agents)
    np.random.shuffle(valid_agents)

    for current_agent in valid_agents: #train_agents
        """ Update Past Trajectories """
        env.start_accumulate_trajs()
        rewards = []
        trs = []
        for _ in range(5):
            ob = env.reset()
            tr = []
            while True:
                action = current_agent.act(ob[None],None,None)[0]
                ob, reward, done, _ = env.step(action)
                tr.append(_['true_reward'])
                if done: break
            trs.append(np.sum(tr))
            rewards.append(reward)
        current_tr = np.mean(trs)
        env.end_accumulate_trajs()

        """ Compare against the current_agent trajectories """
        for evolved_agent in valid_agents:
            rewards = []
            trs = []
            for _ in range(5):
                ob = env.reset()
                tr = []
                while True:
                    action = evolved_agent.act(ob[None],None,None)[0]
                    ob, reward, done, _ = env.step(action)
                    tr.append(_['true_reward'])
                    if done: break
                trs.append(np.sum(tr))
                rewards.append(reward)
            print(current_tr,np.mean(trs),'should be',1 if (current_tr<np.mean(trs)) else -1,np.mean(rewards))
        print('---')

    # Let's plot things / tendency analysis: (diff. in true rewrad) vs (reward from ranker)
