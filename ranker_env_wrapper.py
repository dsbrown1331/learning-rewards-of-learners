import os, sys
import numpy as np
import tensorflow as tf

# Import my own libraries
sys.path.append('./learner/baselines/')

# TF Log Level configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Only show ERROR log

class RankerEnv(object):
    def __init__(self,
                 env_id,
                 env_type,
                 env_params_path,
                 RankerModel,
                 model_path):
        """
        Make Environment
        """
        from baselines.common.cmd_util import make_vec_env
        from baselines.common.vec_env.vec_frame_stack import VecFrameStack
        from baselines.common.vec_env.vec_normalize import VecNormalize

        #env id, env type, num envs, and seed
        env = make_vec_env(env_id, env_type, 1, 0, wrapper_kwargs={'clip_rewards':False,'episode_life':False,})

        if env_type == 'atari':
            self.env = VecFrameStack(env, 4)
        elif env_type == 'mujoco':
            self.env = VecNormalize(env,ob=True,ret=False,eval=True)
            self.env.load(env_params_path)
        else:
            assert False, 'not supported env type'

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
                self.ranker_model = RankerModel()
                saver = tf.train.Saver(var_list=self.ranker_model.parameters(train=False),max_to_keep=0)
                saver.restore(self.sess,model_path)

        """
        Previous & Current Trajectories to evaluate rewards.
        """
        obs = [self.env.reset()]
        while True:
            action = self.env.action_space.sample()
            ob, reward, done, _ = self.env.step(action)
            obs.append(ob)
            if done: break
        self.past_trajs = [np.concatenate(obs,axis=0)]

        self.recent_trajs = []
        self.current_traj = []

    def seed(self,seed):
        return self.env.seed(seed)

    def render(self,mode):
        return self.env.render(mode)

    def reset(self):
        ob = self.env.reset()
        self.current_traj = [ob]

        return ob[0] #Since self.env is DummyVecEnv.

    def step(self,action):
        ob, true_r, done, info = self.env.step(action[None])
        self.current_traj.append(ob)

        if done[0] :
            traj = np.concatenate(self.current_traj,axis=0)
            reward = self.calc_reward(traj)

            self.recent_trajs.append(traj)
        else:
            reward = 0.

        return ob[0], reward, done[0], {'true_reward':true_r}

    def update(self):
        if len(self.recent_trajs) > 0:
            self.past_trajs = self.recent_trajs

    def calc_reward(self,traj,batch_size=16,steps=20):
        b_x, b_y = [], []
        for _ in range(batch_size):
            def _pick_traj_and_crop(trajs):
                idx = np.random.randint(len(trajs))
                obs = trajs[idx]

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

# Test Code
if __name__ == "__main__":
    from siamese_ranker import Model
    from functools import partial

    env = RankerEnv(
        'Hopper-v2',
        'mujoco',
        './learner/models/hopper/checkpoints/00480',
        partial(Model, 11*20, 128), #ob_dim * steps
        './log/Hopper-v2/model.ckpt-25000'
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
        rewards = []
        trs = []
        for _ in range(10):
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

        env.update()
        for evolved_agent in valid_agents:
            rewards = []
            trs = []
            for _ in range(10):
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
