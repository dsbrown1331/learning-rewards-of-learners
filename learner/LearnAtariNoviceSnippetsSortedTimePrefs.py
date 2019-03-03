import argparse
# coding: utf-8

# Take length 50 snippets and record the cumulative return for each one. Then determine ground truth labels based on this.

# In[1]:


import pickle
import gym
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from run_test import *
#import matplotlib.pylab as plt

def normalize_state(obs):
    obs_highs = env.observation_space.high
    obs_lows = env.observation_space.low
    #print(obs_highs)
    #print(obs_lows)
    #return  2.0 * (obs - obs_lows) / (obs_highs - obs_lows) - 1.0
    return obs / 255.0


def mask_score(obs):
    #takes a stack of four observations and blacks out (sets to zero) top n rows
    n = 10
    #no_score_obs = copy.deepcopy(obs)
    obs[:,:n,:,:] = 0
    return obs

def generate_novice_demos(env, env_name, agent, model_dir):
    checkpoint_min = 50
    checkpoint_max = 600
    checkpoint_step = 50
    checkpoints = ['00001']
    if env_name == "enduro":
        checkpoint_min = 3100
        checkpoint_max = 3600
    elif env_name == "seaquest":
        checkpoint_min = 10
        checkpoint_max = 65
        checkpoint_step = 5
    for i in range(checkpoint_min, checkpoint_max + checkpoint_step, checkpoint_step):
        if i < 10:
            checkpoints.append('0000' + str(i))
        elif i < 100:
            checkpoints.append('000' + str(i))
        elif i < 1000:
            checkpoints.append('00' + str(i))
        elif i < 10000:
            checkpoints.append('0' + str(i))
    print(checkpoints)



    demonstrations = []
    learning_returns = []
    learning_rewards = []
    for checkpoint in checkpoints:

        model_path = model_dir + "/models/" + env_name + "_25/" + checkpoint
        if env_name == "seaquest":
            model_path = model_dir + "/models/" + env_name + "_5/" + checkpoint

        agent.load(model_path)
        episode_count = 1
        for i in range(episode_count):
            done = False
            traj = []
            gt_rewards = []
            r = 0

            ob = env.reset()
            #traj.append(ob)
            #print(ob.shape)
            steps = 0
            acc_reward = 0
            while True:
                action = agent.act(ob, r, done)
                ob, r, done, _ = env.step(action)
                #print(ob.shape)
                traj.append(mask_score(normalize_state(ob)))
                gt_rewards.append(r[0])
                steps += 1
                acc_reward += r[0]
                if done:
                    print("checkpoint: {}, steps: {}, return: {}".format(checkpoint, steps,acc_reward))
                    break
            print("traj length", len(traj))
            print("demo length", len(demonstrations))
            demonstrations.append(traj)
            learning_returns.append(acc_reward)
            learning_rewards.append(gt_rewards)

    return demonstrations, learning_returns, learning_rewards






#cheat and sort them to see if it helps learning
#sorted_demos = [x for _, x in sorted(zip(learning_returns,demonstrations), key=lambda pair: pair[0])]

#sorted_returns = sorted(learning_returns)
#print(sorted_returns)
#plt.plot(sorted_returns)


# Create training data by taking random 50 length crops of trajectories, computing the true returns and adding them to the training data with the correct label.
#

# In[9]:

def create_training_data(demonstrations, n_train, snippet_length):
    #n_train = 3000 #number of pairs of trajectories to create
    #snippet_length = 50
    training_obs = []
    training_labels = []
    num_demos = len(demonstrations)
    for n in range(n_train):
        ti = 0
        tj = 0
        #only add trajectories that are different returns
        while(ti == tj):
            #pick two random demonstrations
            ti = np.random.randint(num_demos)
            tj = np.random.randint(num_demos)
            #print(ti, tj)
            #create random snippets
            #find min length of both demos to ensure we can pick a demo no earlier than that chosen in worse preferred demo
            min_length = min(len(demonstrations[ti]), len(demonstrations[tj]))
            if ti < tj: #pick tj snippet to be later than ti
                ti_start = np.random.randint(min_length-snippet_length+1)
                print(ti_start, len(demonstrations[tj]))
                tj_start = np.random.randint(ti_start, len(demonstrations[tj]) - snippet_length + 1)
            else: #ti is better so pick later snippet in ti
                tj_start = np.random.randint(min_length-snippet_length+1)
                print(tj_start, len(demonstrations[ti]))
                ti_start = np.random.randint(tj_start, len(demonstrations[ti]) - snippet_length + 1)
            #print("start", ti_start, tj_start)
            traj_i = demonstrations[ti][ti_start:ti_start+snippet_length]
            traj_j = demonstrations[tj][tj_start:tj_start+snippet_length]
            #print('traj', traj_i, traj_j)
            #return_i = sum(learning_rewards[ti][ti_start:ti_start+snippet_length])
            #return_j = sum(learning_rewards[tj][tj_start:tj_start+snippet_length])
            #print("returns", return_i, return_j)

        #if return_i > return_j:
        #    label = 0
        #else:
        #    label = 1
        if ti > tj:
            label = 0
        else:
            label = 1
        #print(label)
        #TODO: maybe add indifferent label?
        training_obs.append((traj_i, traj_j))
        training_labels.append(label)

    return training_obs, training_labels







class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 16, 7, stride=3)
        self.conv2 = nn.Conv2d(16, 16, 5, stride=2)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1)
        self.fc1 = nn.Linear(784, 64)
        #self.fc1 = nn.Linear(1936,64)
        self.fc2 = nn.Linear(64, 1)



    def cum_return(self, traj):
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0
        sum_abs_rewards = 0
        for x in traj:
            x = x.permute(0,3,1,2) #get into NCHW format
            #compute forward pass of reward network
            x = F.leaky_relu(self.conv1(x))
            x = F.leaky_relu(self.conv2(x))
            x = F.leaky_relu(self.conv3(x))
            x = F.leaky_relu(self.conv4(x))
            x = x.view(-1, 784)
            #x = x.view(-1, 1936)
            x = F.leaky_relu(self.fc1(x))
            #r = torch.tanh(self.fc2(x)) #clip reward?
            r = self.fc2(x)
            sum_rewards += r
            sum_abs_rewards += torch.abs(r)
        ##    y = self.scalar(torch.ones(1))
        ##    sum_rewards += y
        #print(sum_rewards)
        return sum_rewards, sum_abs_rewards



    def forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''
        #print([self.cum_return(traj_i), self.cum_return(traj_j)])
        cum_r_i, abs_r_i = self.cum_return(traj_i)
        cum_r_j, abs_r_j = self.cum_return(traj_j)
        #print(abs_r_i + abs_r_j)
        return torch.cat([cum_r_i, cum_r_j]), abs_r_i + abs_r_j




# Now we train the network. I'm just going to do it one by one for now. Could adapt it for minibatches to get better gradients

# In[111]:


def learn_reward(reward_network, optimizer, training_inputs, training_outputs, num_iter, l1_reg, checkpoint_dir):
    #check if gpu available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)
    loss_criterion = nn.CrossEntropyLoss()
    #print(training_data[0])
    cum_loss = 0.0
    training_data = list(zip(training_inputs, training_outputs))
    for epoch in range(num_iter):
        np.random.shuffle(training_data)
        training_obs, training_labels = zip(*training_data)
        for i in range(len(training_labels)):
            traj_i, traj_j = training_obs[i]
            labels = np.array([[training_labels[i]]])
            traj_i = np.array(traj_i)
            traj_j = np.array(traj_j)
            traj_i = torch.from_numpy(traj_i).float().to(device)
            traj_j = torch.from_numpy(traj_j).float().to(device)
            labels = torch.from_numpy(labels).to(device)

            #zero out gradient
            optimizer.zero_grad()

            #forward + backward + optimize
            outputs, abs_rewards = reward_network.forward(traj_i, traj_j)
            outputs = outputs.unsqueeze(0)
            #print(outputs)
            #print(labels)
            loss = loss_criterion(outputs, labels) + l1_reg * abs_rewards
            loss.backward()
            optimizer.step()

            #print stats to see if learning
            item_loss = loss.item()
            cum_loss += item_loss
            if i % 500 == 499:
                #print(i)
                print("epoch {}:{} loss {}".format(epoch,i, cum_loss))
                print(abs_rewards)
                cum_loss = 0.0
                print("check pointing")
                torch.save(reward_net.state_dict(), checkpoint_dir)
    print("finished training")





def calc_accuracy(reward_network, training_inputs, training_outputs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_criterion = nn.CrossEntropyLoss()
    #print(training_data[0])
    num_correct = 0.
    with torch.no_grad():
        for i in range(len(training_inputs)):
            label = training_outputs[i]
            #print(inputs)
            #print(labels)
            traj_i, traj_j = training_inputs[i]
            traj_i = np.array(traj_i)
            traj_j = np.array(traj_j)
            traj_i = torch.from_numpy(traj_i).float().to(device)
            traj_j = torch.from_numpy(traj_j).float().to(device)

            #forward to get logits
            outputs, abs_return = reward_network.forward(traj_i, traj_j)
            #print(outputs)
            _, pred_label = torch.max(outputs,0)
            #print(pred_label)
            #print(label)
            if pred_label.item() == label:
                num_correct += 1.
    return num_correct / len(training_inputs)






def predict_reward_sequence(net, traj):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rewards_from_obs = []
    with torch.no_grad():
        for s in traj:
            r = net.cum_return(torch.from_numpy(np.array([s])).float().to(device))[0].item()
            rewards_from_obs.append(r)
    return rewards_from_obs

def predict_traj_return(net, traj):
    return sum(predict_reward_sequence(net, traj))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_name', default='', help='Select the environment name to run, i.e. pong')
    parser.add_argument('--reward_model_path', default='', help="name and location for learned model params")
    parser.add_argument('--seed', default=0, help="random seed for experiments")
    parser.add_argument('--models_dir', default = ".", help="top directory where checkpoint models for demos are stored")

    args = parser.parse_args()
    env_name = args.env_name
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
    print(env_type)
    #set seeds
    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    print("Training reward for", env_id)
    n_train = 6000 #number of pairs of trajectories to create
    snippet_length = 50 #length of trajectory for training comparison
    lr = 0.0001
    weight_decay = 0.0
    num_iter = 5
    l1_reg=0.0
    stochastic = True

    #env id, env type, num envs, and seed
    env = make_vec_env(env_id, 'atari', 1, seed,
                       wrapper_kwargs={
                           'clip_rewards':False,
                           'episode_life':False,
                       })


    env = VecFrameStack(env, 4)
    agent = PPO2Agent(env, env_type, stochastic)

    demonstrations, learning_returns, learning_rewards = generate_novice_demos(env, env_name, agent, args.models_dir)
    # Let's plot the returns to see if they are roughly monotonically increasing.
    #plt.plot(learning_returns)
    #plt.xlabel("Demonstration")
    #plt.ylabel("Return")
    #plt.savefig(env_type + "LearningCurvePPO.png")
    #plt.show()

    #sort the demonstrations according to ground truth reward

    print(len(learning_returns))
    print(len(demonstrations))
    print([a[0] for a in zip(learning_returns, demonstrations)])
    #sort them based on human preferences
    demonstrations = [x for _, x in sorted(zip(learning_returns,demonstrations), key=lambda pair: pair[0])]

    sorted_returns = sorted(learning_returns)
    print(sorted_returns)
    #plt.plot(sorted_returns)
    #plt.show()

    training_obs, training_labels = create_training_data(demonstrations, n_train, snippet_length)
    print("num training_obs", len(training_obs))
    print("num_labels", len(training_labels))
    # Now we create a reward network and optimize it using the training data.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reward_net = Net()
    reward_net.to(device)
    import torch.optim as optim
    optimizer = optim.Adam(reward_net.parameters(),  lr=lr, weight_decay=weight_decay)
    learn_reward(reward_net, optimizer, training_obs, training_labels, num_iter, l1_reg, args.reward_model_path)

    with torch.no_grad():
        pred_returns = [predict_traj_return(reward_net, traj) for traj in demonstrations]
    for i, p in enumerate(pred_returns):
        print(i,p,sorted_returns[i])

    print("accuracy", calc_accuracy(reward_net, training_obs, training_labels))


    #TODO:add checkpoints to training process
    torch.save(reward_net.state_dict(), args.reward_model_path)
