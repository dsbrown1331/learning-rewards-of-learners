import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
import gym
from tqdm import tqdm
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from imgcat import imgcat

from tf_commons.ops import *
from siamese_ranker import PPO2Agent

class Model(object):
    def __init__(self,in_dims,embedding_dims=256,steps=40):
        self.ob = tf.placeholder(tf.float32,[None,in_dims])
        self.x = tf.placeholder(tf.float32,[None,steps,in_dims])
        self.y = tf.placeholder(tf.float32,[None,steps,in_dims])
        self.l = tf.placeholder(tf.int32,[None]) # [0 when x is better 1 when y is better]
        self.l2_reg = tf.placeholder(tf.float32,[]) # [0 when x is better 1 when y is better]


        with tf.variable_scope('weights') as param_scope:
            self.fc1 = Linear('fc1',in_dims,embedding_dims)
            self.fc2 = Linear('fc2',embedding_dims,embedding_dims)
            self.fc3 = Linear('fc3',embedding_dims,1)

        self.param_scope = param_scope

        # build graph
        def _reward(x):
            _ = tf.nn.relu(self.fc1(x))
            _ = tf.nn.relu(self.fc2(_))
            r = tf.squeeze(self.fc3(_),axis=1)
            return r

        self.r = _reward(self.ob)

        self.r_x = tf.reshape(_reward(tf.reshape(self.x,[-1,in_dims])),[-1,steps])
        self.sum_r_x = tf.reduce_sum(self.r_x,axis=1)

        self.r_y = tf.reshape(_reward(tf.reshape(self.y,[-1,in_dims])),[-1,steps])
        self.sum_r_y = tf.reduce_sum(self.r_y,axis=1)

        logits = tf.stack([self.sum_r_x,self.sum_r_y],axis=1) #[None,2]
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=self.l)
        self.loss = tf.reduce_mean(loss,axis=0)

        weight_decay = tf.reduce_sum(self.fc1.w**2) + tf.reduce_sum(self.fc2.w**2) + tf.reduce_sum(self.fc3.w**2)
        self.l2_loss = self.l2_reg * weight_decay

        pred = tf.cast(tf.greater(self.sum_r_y,self.sum_r_x),tf.int32)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(pred,self.l),tf.float32))

        self.optim = tf.train.AdamOptimizer(1e-4)
        self.update_op = self.optim.minimize(self.loss+self.l2_loss,var_list=self.parameters(train=True))

        self.saver = tf.train.Saver(var_list=self.parameters(train=False),max_to_keep=0)

    def parameters(self,train=False):
        if train:
            return tf.trainable_variables(self.param_scope.name)
        else:
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,self.param_scope.name)

    def train(self,D,batch_size=64,iter=10000,l2_reg=0.01,debug=False):
        """
        Training will be early-terminate when validation accuracy becomes large enough..?

        args:
            D: list of triplets (\sigma^1,\sigma^2,\mu)
            while
                sigma^{1,2}: shape of [steps,in_dims]
                mu : 0 or 1
        """
        sess = tf.get_default_session()

        idxes = np.random.permutation(len(D))
        train_idxes = idxes[:int(len(D)*0.8)]
        valid_idxes = idxes[int(len(D)*0.8):]

        def _batch(idx_list,add_noise):
            batch = []

            if len(idx_list) > batch_size:
                idxes = np.random.choice(idx_list,batch_size,replace=False)
            else:
                idxes = idx_list

            for i in idxes:
                batch.append(D[i])

            b_x,b_y,b_l = zip(*batch)
            b_x,b_y,b_l = np.array(b_x),np.array(b_y),np.array(b_l)

            if add_noise:
                b_l = (b_l + np.random.binomial(1,0.1,batch_size)) % 2 #Flip it with probability 0.1

            return b_x,b_y,b_l

        for it in tqdm(range(iter)):
            b_x,b_y,b_l = _batch(train_idxes,add_noise=True)
            loss,l2_loss,acc,_ = sess.run([self.loss,self.l2_loss,self.acc,self.update_op],feed_dict={
                self.x:b_x,
                self.y:b_y,
                self.l:b_l,
                self.l2_reg:l2_reg,
            })

            b_x,b_y,b_l = _batch(valid_idxes,add_noise=False)
            valid_acc = sess.run(self.acc,feed_dict={
                self.x:b_x,
                self.y:b_y,
                self.l:b_l
            })

            if debug:
                if it % 100 == 0 or it < 10:
                    tqdm.write(('loss: %f (l2_loss: %f), acc: %f, valid_acc: %f'%(loss,l2_loss,acc,valid_acc)))

            #if valid_acc >= 0.95:
            #    print('loss: %f (l2_loss: %f), acc: %f, valid_acc: %f'%(loss,l2_loss,acc,valid_acc))
            #    print('early termination@%08d'%it)
            #    break

    def eval(self,D,batch_size=64):
        sess = tf.get_default_session()

        b_x,b_y,b_l = zip(*D)
        b_x,b_y,b_l = np.array(b_x),np.array(b_y),np.array(b_l)

        b_r_x, b_acc = [], 0.

        for i in range(0,len(b_x),batch_size):
            sum_r_x, acc = sess.run([self.sum_r_x,self.acc],feed_dict={
                self.x:b_x[i:i+batch_size],
                self.y:b_y[i:i+batch_size],
                self.l:b_l[i:i+batch_size]
            })

            b_r_x.append(sum_r_x)
            b_acc += len(sum_r_x)*acc

        return np.concatenate(b_r_x,axis=0), b_acc/len(b_x)

    def get_reward(self,obs,batch_size=1024):
        sess = tf.get_default_session()

        b_r = []
        for i in range(0,len(obs),batch_size):
            r = sess.run(self.r,feed_dict={
                self.ob:obs[i:i+batch_size],
            })

            b_r.append(r)

        return np.concatenate(b_r,axis=0)

class GTDataset(object):
    def __init__(self,env):
        self.env = env

    def gen_traj(self,agent):
        obs, actions, rewards = [self.env.reset()], [], [0.]
        while True:
            action = agent.act(obs[-1], rewards[-1], False)
            ob, reward, done, _ = self.env.step(action)

            obs.append(ob)
            actions.append(action)
            rewards.append(reward)

            if done: break

        return np.stack(obs,axis=0), np.array(actions), np.array(rewards)

    def prebuilt(self,agents,num_trajs):
        trajs = []
        for agent in tqdm(agents):
            for _ in range(num_trajs):
                traj = self.gen_traj(agent)
                #print('model', agent.model_path, 'reward:',np.sum(traj[2]))

                trajs.append(traj)
                tqdm.write('model: %s avg reward: %f'%(agent.model_path,np.mean([np.sum(traj[2]) for traj in trajs])))
        obs,actions,rewards = zip(*trajs)
        self.trajs = (np.concatenate(obs,axis=0),np.concatenate(rewards,axis=0))

        print(self.trajs[0].shape,self.trajs[1].shape)

    def sample(self,num_samples,steps=40):
        obs, rewards = self.trajs

        D = []
        for _ in range(num_samples):
            x_ptr = np.random.randint(len(obs)-steps)
            y_ptr = np.random.randint(len(obs)-steps)

            D.append( (obs[x_ptr:x_ptr+steps],
                       obs[y_ptr:y_ptr+steps],
                       0 if np.sum(rewards[x_ptr:x_ptr+steps]) > np.sum(rewards[y_ptr:y_ptr+steps]) else 1
                       ))
        return D

def train(args):
    logdir = Path(args.logbase_path) / args.env_id
    if logdir.exists() :
        c = input('log is already exist. continue [Y/etc]? ')
        if c in ['YES','yes','Y']:
            import shutil
            shutil.rmtree(str(logdir))
        else:
            print('good bye')
            exit()
    logdir.mkdir(parents=True)
    logdir = str(logdir)
    env = gym.make(args.env_id)

    train_agents = []
    valid_agents = []
    test_agents = []

    models = sorted(Path(args.learners_path).glob('?????'))
    for i,path in enumerate(models):
        agent = PPO2Agent(env,args.env_type,str(path))

        if i < len(models)*0.5:
            if i % 2 == 0 :
                train_agents.append(agent)
        else:
            break

    gt_dataset = GTDataset(env)
    gt_dataset.prebuilt(train_agents,1)

    models = []
    for i in range(args.num_models):
        with tf.variable_scope('model_%d'%i):
            models.append(Model(env.observation_space.shape[0]))

    ### Initialize Parameters
    init_op = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
    # Training configuration
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession()

    sess.run(init_op)

    for i,model in enumerate(models):
        D = gt_dataset.sample(args.D)
        model.train(D,l2_reg=args.l2_reg,debug=True)

        model.saver.save(sess,logdir+'/model_%d.ckpt'%(i),write_meta_graph=False)

def eval(args):
    logdir = str(Path(args.logbase_path) / args.env_id)

    env = gym.make(args.env_id)

    valid_agents = []
    test_agents = []

    models = sorted(Path(args.learners_path).glob('?????'))
    for i,path in enumerate(models):
        agent = PPO2Agent(env,args.env_type,str(path))

        if i < len(models)*0.5:
            if i % 2 != 0 :
                valid_agents.append(agent)
        else:
            test_agents.append(agent)

    gt_dataset= GTDataset(env)
    gt_dataset.prebuilt(valid_agents,1)
    D_valid = gt_dataset.sample(args.D)

    gt_dataset_test = GTDataset(env)
    gt_dataset_test.prebuilt(test_agents,1)
    D_test = gt_dataset_test.sample(args.D)

    models = []
    for i in range(args.num_models):
        with tf.variable_scope('model_%d'%i):
            models.append(Model(env.observation_space.shape[0]))

    ### Initialize Parameters
    init_op = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
    # Training configuration
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession()

    sess.run(init_op)

    for i,model in enumerate(models):
        model.saver.restore(sess,logdir+'/model_%d.ckpt'%(i))

        print('model %d'%i)

        reward_sum, acc = model.eval(D_valid)
        print('valid_dataset', np.mean(reward_sum),np.std(reward_sum),acc)

        reward_sum, acc = model.eval(D_test)
        print('test_datset', np.mean(reward_sum),np.std(reward_sum),acc)

        obs, r = gt_dataset.trajs
        r_hat = model.get_reward(obs)

        obs, r_test = gt_dataset_test.trajs
        r_hat_test = model.get_reward(obs)

        fig,axes = plt.subplots(1,2)
        axes[0].plot(r,r_hat,'o')
        axes[1].plot(r_test,r_hat_test,'o')
        fig.savefig('model_%d.png'%i)
        imgcat(fig)
        plt.close(fig)

        np.savez('model_%d.npz'%i,r=r,r_hat=r_hat,r_test=r_test,r_hat_test=r_hat_test)



if __name__ == "__main__":
    # Required Args (target envs & learners)
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_id', default='', help='Select the environment to run')
    parser.add_argument('--env_type', default='', help='mujoco or atari')
    parser.add_argument('--learners_path', default='', help='path of learning agents')
    parser.add_argument('--num_models', default=3, type=int, help='number of models to ensemble')
    parser.add_argument('--l2_reg', default=0.01, type=float, help='l2 regularization size')
    parser.add_argument('--D', default=1000, type=int, help='|D| in the preference paper')
    parser.add_argument('--logbase_path', default='./log_preference/', help='path to log base (env_id will be concatenated at the end)')

    parser.add_argument('--eval', action='store_true', help='path to log base (env_id will be concatenated at the end)')
    args = parser.parse_args()

    if not args.eval :
        train(args)
    else:
        eval(args)
