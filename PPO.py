'''
Actor Critic Methods will often have performance tanks after a certain amount
of time due to being sensitive to perturbations. 
This was the inspiration behind the PPO algorithm. Effectively the process
of making the TRPO algorithm more efficient and less prone to mass fluctuations.

It does this by using what the paper calls 'clipped probability ratios'
which is effectively comparing policies between timesteps to eachother 
with a set lower bound. Basing the update of the policy between some 
ratio of a new policy to the old. The term probability comes due to having
0-1 as bounds.

PPO also keeps 'memories' maybe similar to that of DQN. Multiple updates 
to the network happen per data sample, which are carried out through
minibatch stochastic gradient ascent. 

Implementation notes: Memory
We note that learning in this case is carried out through batches. 
We keep a track of, say, 50 state transitions, then train on a batch 
of 5-10-15 of them. The size of the batch is arbitrary for implementation 
but there likely exists a best batch size. It seems to be the case that 
the batches are carried out from iterative state transfers only. 

Implementation notes: Critic
Two distinct networks instead of shared inputs. 
Actor critic methods do not note state action pairs, just states. 
Actor decides to do based on the current state, and the critic evaluates states.

Critic Loss:
Return = advantage + critic value (from memory).
then the L_critic = MSE(return - critic vlaue (from network))

Networks outputs probabilities for an action distribution, therefore exploration is
handled by definition. 

Overview:
Class for replay buffer, which can be implemented quite well with lists. 
Class for actor network and critic network
Class for the agent, tying everything together
Main loop to train and evaluate

'''

import os 
import numpy as np
import torch
import torch.distributions as dist
from collections import namedtuple, deque
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
import typing


############ HYPERPARAMS #########
CRITIC_LR = 0.01
ACTOR_LR = 0.0001

GAMMA = 0.95
CLIP = 0.2
TIMESTEPS = 110
MINIBATCH_SIZE = 12

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward')) #saving the result of taking action a in state s, we progress to the next state and observe a reward


class ActorCritic(nn.Module):
    # We create a different class for our ACTOR-CRITICS so we can save old_policies. 
    def __init__(self, input_dims, output_dims):
        super(ActorCritic, self).__init__()
  
        self.actor = nn.Sequential(
            nn.Linear(input_dims, 215),
            nn.ReLU(),
            nn.Linear(215, 215),
            nn.ReLU(),
            nn.Linear(215, output_dims),
            nn.Softmax()
        )

        self.critic = nn.Sequential(
            nn.Linear(input_dims, 214),
            nn.ReLU(),
            nn.Linear(214, 94),
            nn.ReLU(),
            nn.Linear(94, 1)
        )

    def forward(self, state):
        '''
        Outputs the distribution functions of log of the policy pi(a|s), as well as 
        the critics state distributions. 
        Also outputs the actual action sampled. 
        '''
        # this outputs the distribution functions

        # We can return the actual action distribution, the log-action distribution, and the critic
        # distribution.
        # probs = self.actor(state)
        action_distribution = Categorical(self.actor(state.flatten()))
        
        # Samples action from the action distribution.
        action = action_distribution.sample()
        log_action_dist = action_distribution.log_prob(action) #(?)

        # Entropy speaks to how sure the network thinks the action is. 
        act_dist_entropy = action_distribution.entropy()

        # Critic evalutation of the network. 

        critic_state_distribution = self.critic(state.flatten())

        # critic_state_distribution /= critic_state_distribution.sum()

        # NOTE TO SELF
        # Experiement with log_action_dist vs action_distribution

        # print("what is this: ", action)
        return action.item(), log_action_dist, critic_state_distribution, act_dist_entropy, action_distribution



class PPO(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(PPO, self).__init__()

        self.policy = ActorCritic(input_dims, output_dims)
        self.old_policy = self.policy

        # Batch states
        # obtained before action made
        self.batch_states = []
        self.policy_action_distributions = []
        self.old_policy_action_distributions = []
        self.value_functions = []                           # critic distributions 
        self.batch_actions = []
        # obtained after 
        self.batch_rewards = []
        self.terminal_state = []

        self.optimizer = torch.optim.Adam([
                {'params': self.policy.actor.parameters(), 'lr': ACTOR_LR},
                {'params': self.policy.critic.parameters(), 'lr': CRITIC_LR}
            ])

    def learn(self):
        '''
        We note that the loss function for PPO takes the combined loss functions 
        of the clipped lost function, the value function loss function, and finally
        an entropy bonus.

        We note that the loss function for the actor and the critic is different. The loss^clip is the loss function for the 
        actor model, which outputs scalar values that correspond to states. 

        v1: learning rate will only consist of the clipping loss.
        v2: learning rate will consist of clipping loss plus an entropy bonus. 

        '''
        print("TERMINAL STATE? ", self.terminal_state)
        print("LENGTH OF TERM STATE: ", len(self.terminal_state))
        # Skip training if batch_size not reached. 
        if len(self.batch_rewards) < MINIBATCH_SIZE:
            return
        
        # set old policy to current policy, before learning
        temp_storage = self.policy.state_dict().copy()

        # get advantage function
        advantage = self.compute_advantage_function(self.value_functions, self.batch_rewards)

        # calculate r_t
        prob_ratio = torch.Tensor(np.divide(np.array(self.policy_action_distributions), np.array(self.old_policy_action_distributions)))

        # Compute probability ratio tensor. 
        prob_ratio = self.clip_ratio(prob_ratio)

        # compute the clipped_loss
        clipped_loss = torch.multiply(prob_ratio, advantage)

        # compute expected value of clipped loss
        loss = torch.mean(clipped_loss)
        loss = loss.requires_grad_(True)

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        # self.old_policy.load_state_dict(temp_storage)

        # clear all our memory
        self.batch_states = []
        self.policy_action_distributions = []
        self.old_policy_action_distributions = []
        self.value_functions = []
        self.batch_actions = []
        self.batch_rewards = []
        self.terminal_state = []
        
    def make_action(self, state: typing.Dict):
        ''' This is different from the actor-critic networks action as we need to store the values we 
            obtain from this'''
        
        input_state = observation_to_values(state)

        with torch.no_grad():
            # Run network to get nessessary variables, while ensuring pytorch does not compute gradients.
            input_state = torch.Tensor(input_state).to(device)
            action, log_action_distribution, critic_distribution, entropy, cur_act_dist = self.policy.forward(input_state)
            old_action, log_old_actor_distribution, old_critic_distribution, old_entropy, old_act_dist = self.old_policy.forward(input_state)

        # Store the nessessary variables.

        print("CURRENT ACTION DIST: ", cur_act_dist.probs, "OLD_ACT_DIST: ", old_act_dist.probs)

        self.batch_states.append(input_state)
        self.policy_action_distributions.append(log_action_distribution)
        self.old_policy_action_distributions.append(log_old_actor_distribution)
        self.value_functions.append(critic_distribution)
        self.batch_actions.append(action)
        self.terminal_state.append(0)

        return action

    def clip_ratio(self, prob_ratio):
        clipped_ratio = torch.clamp(torch.Tensor(prob_ratio), 1 - CLIP, 1 + CLIP)
        return torch.min(prob_ratio, clipped_ratio)

    def compute_advantage_function(self, states, rewards):
        '''
        params:
            states: an array of self.value_functions
            reward: the tensor of self.batch_rewards
            new_states: the tensor of self.batch_new_states

        This is the calculation of the advantage function.

        We can derive the advantage function in the paper to be equivilant to

        Using this, we can do the following:
        '''


        # best_action, _ = torch.max(self.critic_distributions, dim=0)
        length_of_rewards = len(rewards)
        indices = np.arange(length_of_rewards).reshape(-1, 1)
        weights = np.power(GAMMA, indices)
        print(weights)
        print("rewards" , rewards)

        # we now calculated our weighted states and rewards. 
        weighted_rewards = rewards * weights.transpose()
        print(weighted_rewards)

        last_state = states[-1] * weights[-1]

        print("sum of weighted rewards ", np.sum(weighted_rewards))
        print(states[0].item())
        print(last_state.item())

        # calculate advantage
        advantage = np.sum(weighted_rewards) - states[0].item() + last_state.item()

        return advantage
        
    def save(self, chkpt):
        torch.save(self.policy_old.state_dict(), chkpt)

    def load(self, chkpt):
        self.old_policy.load_state_dict(torch.load(chkpt, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(chkpt, map_location=lambda storage, loc: storage))

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the AdamW optimizer
BATCH_SIZE = 500
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.001 #in long games each action is really important, so we want to be greedy after lots of training
EPS_DECAY = 1000

def observation_to_values(observation):
    # Nathan's modified observation helper function
    #init
    board = observation['board']
    #print(board)
    health = 100
    n_channels = 5
    state_matrix = np.zeros((n_channels, board["height"], board["width"]))
    #fill
    for _snake in board['snakes']:
        if _snake['id'] == '92addb3f-1e17-483f-90bc-e3a386b7a92a':
            body_length = _snake['length']
        health = np.array(_snake['health'])
        #place head on channel 0
        state_matrix[0, _snake['head']['x'], _snake['head']['y']] = 1
        #place tail on channel 1
        state_matrix[1, _snake['body'][-1]['x'], _snake['body'][-1]['y']] = 1
        #place body on channel 2
        for _body_segment in _snake['body']:
            state_matrix[2, _body_segment['x'], _body_segment['y']] = 1
        
    # Calculate variables nessessary for reward.
    num_snakes = np.count_nonzero(state_matrix[0])
    #body_length = np.count_nonzero(state_matrix[2])
    print("body length: ", body_length)


    #place food on channel 3
    for _food in board["food"]:
        state_matrix[3,_food['x'], _food['y']] = 1
    #create health channel
    state_matrix[4] = np.full((board["height"], board["width"]), health)
    #flatten
    # state_matrix = state_matrix.reshape(-1,1) # don't flatten if using conv layer

    state_matrix = state_matrix.flatten()
    #print()
    print(state_matrix.shape)
    state_matrix = np.append(state_matrix, [num_snakes, body_length])
    print(state_matrix.shape)

    #state_matrix.insert(health, num_snakes, body_length)
    
    return state_matrix

# testing
if __name__ == '__main__':
    agent = PPO(365, 4)
    random_1 = torch.rand(365)
    for i in range(17):
        print(agent.make_action(random_1))

    agent.learn()