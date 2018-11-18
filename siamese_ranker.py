import os, sys
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Import my own libraries
sys.path.append('./learner/baselines/')
from tf_commons.ops import *

# TF Log Level configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Only show ERROR log
EMBEDDING_DIMS = 128

class Model(object):
    def __init__(self,in_dims,embedding_dims):
        self.x = tf.placeholder(tf.float32,[None,in_dims])
        self.y = tf.placeholder(tf.float32,[None,in_dims])
        self.l = tf.placeholder(tf.int32,[None,1])

        with tf.variable_scope('weights') as param_scope:
            self.embed_fc1 = Linear('embed_fc1',in_dims,embedding_dims)
            self.embed_fc2 = Linear('embed_fc2',embedding_dims,embedding_dims)
            self.embed_fc3 = Linear('embed_fc3',embedding_dims,embedding_dims)

            self.rank_fc1 = Linear('rank_fc1',embedding_dims*2,embedding_dims)
            self.rank_fc2 = Linear('rank_fc2',embedding_dims,embedding_dims)
            self.rank_fc3 = Linear('rank_fc3',embedding_dims,1)

        self.param_scope = param_scope

        # build graph
        def _embed(x):
            _ = tf.nn.relu(self.embed_fc1(x))
            _ = tf.nn.relu(self.embed_fc2(_))
            embedding = self.embed_fc3(_)
            return embedding

        self.e_x = _embed(self.x)
        self.e_y = _embed(self.y)

        _ = tf.concat([self.e_x,self.e_y],axis=1)
        _ = tf.nn.relu(self.rank_fc1(_))
        _ = tf.nn.relu(self.rank_fc2(_))
        self.logits = self.rank_fc3(_)

        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,labels=tf.cast(self.l,tf.float32)))
        self.pred = tf.greater(tf.sigmoid(self.logits),0.5)
        self.acc = tf.count_nonzero(tf.equal(self.pred,tf.cast(self.l,tf.bool)),dtype=tf.float64) / tf.cast(tf.shape(self.logits)[0],tf.float64)

    def parameters(self,train=False):
        if train:
            return tf.trainable_variables(self.param_scope.name)
        else:
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,self.param_scope.name)

class PPO2Agent(object):
    def __init__(self, env, env_type, path, gpu=True):
        from baselines.common.policies import build_policy
        from baselines.ppo2.model import Model

        self.graph = tf.Graph()

        if gpu:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
        else:
            config = tf.ConfigProto(device_count = {'GPU': 0})

        self.sess = tf.Session(graph=self.graph,config=config)

        with self.graph.as_default():
            with self.sess.as_default():
                ob_space = env.observation_space
                ac_space = env.action_space

                if env_type == 'atari':
                    policy = build_policy(env,'cnn')
                elif env_type == 'mujoco':
                    policy = build_policy(env,'mlp')
                else:
                    assert False,' not supported env_type'

                make_model = lambda : Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=1, nbatch_train=1,
                                nsteps=1, ent_coef=0., vf_coef=0.,
                                max_grad_norm=0.)
                self.model = make_model()

                self.model_path = path
                self.model.load(path)

    def act(self, observation, reward, done):
        a,v,state,neglogp = self.model.step(observation)
        return a

class Dataset(object):
    def __init__(self,env_id,env_type):
        from baselines.common.cmd_util import make_vec_env
        from baselines.common.vec_env.vec_frame_stack import VecFrameStack
        from baselines.common.vec_env.vec_normalize import VecNormalize

        #env id, env type, num envs, and seed
        env = make_vec_env(env_id, env_type, 1, 0, wrapper_kwargs={'clip_rewards':False,'episode_life':False,})

        if env_type == 'atari':
            self.env = VecFrameStack(env, 4)
        elif env_type == 'mujoco':
            self.env = VecNormalize(env,ob=True,ret=False,eval=True)
        else:
            assert False, 'not supported env type'

    def gen_traj(self,agent):
        try:
            self.env.load(agent.model_path)
        except AttributeError:
            pass

        obs, actions, rewards = [self.env.reset()], [], [0.]
        while True:
            action = agent.act(obs[-1], rewards[-1], False)
            ob, reward, done, _ = self.env.step(action)

            obs.append(ob)
            actions.append(action)
            rewards.append(reward)

            if done: break

        return np.concatenate(obs,axis=0), np.array(actions), np.array(rewards)

    def prebuilt(self,agents,num_trajs):
        ranked_trajs = []
        for agent in tqdm(agents):
            trajs = []
            for _ in range(num_trajs):
                traj = self.gen_traj(agent)
                #print('model', agent.model_path, 'reward:',np.sum(traj[2]))

                trajs.append(traj)
            tqdm.write('model: %s avg reward: %f'%(
                agent.model_path,
                np.mean([np.sum(traj[2]) for traj in trajs])))
            ranked_trajs.append(trajs)
        self.ranked_trajs = ranked_trajs

    def batch(self,batch_size,steps):
        b_x, b_y, b_l = [], [], []
        for _ in range(batch_size):
            x_learner_idx, y_learner_idx = np.random.choice(len(self.ranked_trajs),2,replace=False)

            def _pick_traj_and_crop(trajs):
                idx = np.random.randint(len(trajs))
                states, actions, rewards = trajs[idx]

                if len(states)-steps <= 0: # Trajectories are too short.
                    _ = np.tile(states[-1:],(steps-len(states),1))
                    return np.concatenate([states,np.tile(states[-1:],(steps-len(states),1))],axis=0)

                t = np.random.randint(len(states)-steps)
                return states[t:t+steps]

            x = _pick_traj_and_crop(self.ranked_trajs[x_learner_idx])
            y = _pick_traj_and_crop(self.ranked_trajs[y_learner_idx])

            b_x.append(x.reshape(-1))
            b_y.append(y.reshape(-1))
            b_l.append(x_learner_idx > y_learner_idx)

        return np.array(b_x).astype(np.float32), np.array(b_y).astype(np.float32), np.array(b_l).astype(np.int32)[:,None]

if __name__ == "__main__":
    # Required Args (target envs & learners)
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_id', default='', help='Select the environment to run')
    parser.add_argument('--env_type', default='', help='mujoco or atari')
    parser.add_argument('--learners_path', default='', help='path of learning agents')
    # Trainig Args
    parser.add_argument('--num_iter', default=30000, help='number of iterations')
    parser.add_argument('--batch_size', default=64, help='batch size')
    parser.add_argument('--lr', default=1e-4, help='learning rate')
    parser.add_argument('--steps', default=20, help='trajectory cropping size')
    parser.add_argument('--num_trajs', default=100, help='# of training trajectories')
    parser.add_argument('--logbase_path', default='./log/', help='path to log base (env_id will be concatenated at the end)')
    args = parser.parse_args()

    logdir = Path(args.logbase_path) / args.env_id
    if logdir.exists() :
        c = input('log is already exist. continue [Y/etc]? ')
        if c in ['YES','yes','Y']:
            import shutil
            shutil.rmtree(str(logdir))
        else:
            print('good bye')
            exit()
    logdir = str(logdir)

    dataset = Dataset(args.env_id,args.env_type)
    valid_dataset = Dataset(args.env_id,args.env_type)

    train_agents = []
    valid_agents = []
    p = Path(args.learners_path)
    for i,path in enumerate(sorted(Path(args.learners_path).glob('?????'))):
        agent = PPO2Agent(dataset.env,args.env_type,str(path))
        if i % 2 == 0 :
            train_agents.append(agent)
        else:
            valid_agents.append(agent)

    print('generate training trajectories...')
    dataset.prebuilt(train_agents,args.num_trajs)
    valid_dataset.prebuilt(valid_agents,args.num_trajs)

    # Separate graph from ppo
    graph = tf.Graph()
    with graph.as_default():
        model = Model(dataset.env.observation_space.shape[0]*args.steps,EMBEDDING_DIMS)
        saver = tf.train.Saver(var_list=model.parameters(train=False),max_to_keep=0)

        optim = tf.train.AdamOptimizer(args.lr)
        update_op = optim.minimize(model.loss,var_list=model.parameters(train=True))

        summary_op = tf.summary.merge([
            tf.summary.scalar('loss/train',model.loss),
            tf.summary.scalar('acc/train',model.acc),
        ])
        valid_summary_op= tf.summary.merge([
            tf.summary.scalar('loss/valid',model.loss),
            tf.summary.scalar('acc/valid',model.acc),
        ])

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

    # Training configuration
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(graph=graph,config=config)
    summary_writer = tf.summary.FileWriter(logdir,sess.graph)

    print('start train!')
    sess.run(init_op)
    try:
        for it in tqdm(range(args.num_iter),dynamic_ncols=True):
            b_x, b_y, b_l = dataset.batch(args.batch_size,args.steps)

            loss, acc, summary_str, _ = sess.run(
                [model.loss,model.acc,summary_op,update_op],
                feed_dict={model.x : b_x,
                           model.y : b_y,
                           model.l : b_l})

            if it % 10 == 0:
                summary_writer.add_summary(summary_str,it)

            if it % 100 == 0:
                b_x, b_y, b_l = valid_dataset.batch(args.batch_size,args.steps)
                valid_loss, valid_acc, summary_str = sess.run(
                    [model.loss,model.acc,valid_summary_op],
                    feed_dict={model.x : b_x,
                               model.y : b_y,
                               model.l : b_l})

                summary_writer.add_summary(summary_str,it)
                tqdm.write('[%5d] %f(%f) %f(%f)'%(it,loss,acc,valid_loss,valid_acc))

            if it % 5000 == 0:
                saver.save(sess,logdir+'/model.ckpt',global_step=it,write_meta_graph=False)
    except KeyboardInterrupt:
        pass
    finally:
        saver.save(sess,logdir+'/last.ckpt',write_meta_graph=False)
