import argparse
import agc.dataset as ds
# coding: utf-8

# Take full trajectories and learn reward

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
import agc_demos
from baselines.common.trex_utils import preprocess





#cheat and sort them to see if it helps learning
#sorted_demos = [x for _, x in sorted(zip(learning_returns,demonstrations), key=lambda pair: pair[0])]

#sorted_returns = sorted(learning_returns)
#print(sorted_returns)
#plt.plot(sorted_returns)


# Create training data by taking random 50 length crops of trajectories, computing the true returns and adding them to the training data with the correct label.
#

# In[9]:



def create_training_data(demonstrations):
    #n_train = 3000 #number of pairs of trajectories to create
    #snippet_length = 50
    training_obs = []
    training_labels = []
    num_demos = len(demonstrations)
    for i in range(num_demos):
        for j in range(i+1,num_demos):
            print(i,j)
            traj_i = demonstrations[i]
            traj_j = demonstrations[j]
            label = 1
            training_obs.append((traj_i, traj_j))
            training_labels.append(label)

    return training_obs, training_labels





# def create_training_data(demonstrations, n_train):
#     #n_train = 3000 #number of pairs of trajectories to create
#     #snippet_length = 50
#     training_obs = []
#     training_labels = []
#     num_demos = len(demonstrations)
#     for n in range(n_train):
#         ti = 0
#         tj = 0
#         #only add trajectories that are different returns
#         while(ti == tj):
#             #pick two random demonstrations
#             ti = np.random.randint(num_demos)
#             tj = np.random.randint(num_demos)
#             #print(ti, tj)
#             #create random snippets
#             #find min length of both demos to ensure we can pick a demo no earlier than that chosen in worse preferred demo
#             #min_length = min(len(demonstrations[ti]), len(demonstrations[tj]))
#             #if ti < tj: #pick tj snippet to be later than ti
#             #    ti_start = np.random.randint(min_length-snippet_length+1)
#             #    #print(ti_start, len(demonstrations[tj]))
#             #    tj_start = np.random.randint(ti_start, len(demonstrations[tj]) - snippet_length + 1)
#             #else: #ti is better so pick later snippet in ti
#             #    tj_start = np.random.randint(min_length-snippet_length+1)
#             #    #print(tj_start, len(demonstrations[ti]))
#             #    ti_start = np.random.randint(tj_start, len(demonstrations[ti]) - snippet_length + 1)
#             #print("start", ti_start, tj_start)
#             traj_i = demonstrations[ti]
#             traj_j = demonstrations[tj]
#             #print('traj', traj_i, traj_j)
#             #return_i = sum(learning_rewards[ti][ti_start:ti_start+snippet_length])
#             #return_j = sum(learning_rewards[tj][tj_start:tj_start+snippet_length])
#             #print("returns", return_i, return_j)
#
#         #if return_i > return_j:
#         #    label = 0
#         #else:
#         #    label = 1
#         if ti > tj:
#             label = 0
#         else:
#             label = 1
#         #print(label)
#         #TODO: maybe add indifferent label?
#         training_obs.append((traj_i, traj_j))
#         training_labels.append(label)
#
#     return training_obs, training_labels







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
            if i % 10 == 9:
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
    parser.add_argument('--data_dir', help="where agc data is located")

    args = parser.parse_args()
    env_name = args.env_name
    if env_name == "spaceinvaders":
        env_id = "SpaceInvadersNoFrameskip-v4"
        agc_env_name =  "spaceinvaders"
    elif env_name == "mspacman":
        env_id = "MsPacmanNoFrameskip-v4"
        agc_env_name = "mspacman"
    elif env_name == "videopinball":
        env_id = "VideoPinballNoFrameskip-v4"
        agc_env_name = "pinball"
    elif env_name == "montezumarevenge":
        env_id = "MontezumaRevengeNoFrameskip-v4"
        agc_env_name = "revenge"
    elif env_name == "qbert":
        env_id = "QbertNoFrameskip-v4"
        agc_env_name = "qbert"
    else:
        print("env_name not supported")
        sys.exit(1)


    env_type = "atari"
    print(env_type)
    #set seeds
    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    print("Training reward for", env_id)
    n_train = 200 #number of pairs of trajectories to create
    snippet_length = 50 #length of trajectory for training comparison
    lr = 0.0001
    weight_decay = 0.001
    num_iter = 20 #num times through training data
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

    data_dir = args.data_dir
    dataset = ds.AtariDataset(data_dir)
    demonstrations, learning_returns = agc_demos.get_preprocessed_trajectories(agc_env_name, dataset, data_dir)

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

    training_obs, training_labels = create_training_data(demonstrations)
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
