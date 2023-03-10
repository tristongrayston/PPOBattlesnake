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


############ HYPERPARAMS #########
CRITIC_LR = 0.01
ACTOR_LR = 0.0001

GAMMA = 0.99
CLIP = 0.2
TIMESTEPS = 110
MINIBATCH_SIZE = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward')) #saving the result of taking action a in state s, we progress to the next state and observe a reward


class ActorCritic(nn.Module):
    # We create a different class for our ACTOR-CRITICS so we can save old_policies. 
    def __init__(self, input_dims, output_dims):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(input_dims, 124),
            nn.ReLU(),
            nn.Linear(124, 64),
            nn.ReLU(),
            nn.Linear(64, output_dims),
            nn.ReLU()
        )

        self.critic = nn.Sequential(
            nn.Linear(input_dims, 124),
            nn.ReLU(),
            nn.Linear(124, 64),
            nn.ReLU(),
            nn.Linear(64, output_dims),
            nn.ReLU()
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
        action_distribution = Categorical(self.actor(state))
        print(action_distribution)
        
        # Samples action from the action distribution.
        action = action_distribution.sample()
        log_action_dist = action_distribution.log_prob(action) #(?)

        # Entropy speaks to how sure the network thinks the action is. 
        act_dist_entropy = action_distribution.entropy()

        # Critic evalutation of the network. 
        critic_state_distribution = self.critic(state)

        # critic_state_distribution /= critic_state_distribution.sum()

        # NOTE TO SELF
        # Experiement with log_action_dist vs action_distribution

        return action, log_action_dist, critic_state_distribution, act_dist_entropy



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
        self.critic_distributions = []
        self.batch_actions = []
        # obtained after 
        self.batch_rewards = []
        self.batch_new_states = []
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
        # Skip training if batch_size not reached. 
        if len(self.batch_states) < MINIBATCH_SIZE:
            pass

        # we have a bunch of lists that we need to convert to tensors.

        print("rewards", self.batch_rewards)
        batch_states = torch.tensor(np.array(self.batch_states), dtype=torch.float32)
        policy_action_distributions = tuple(self.policy_action_distributions)
        old_policy_action_distributions = tuple(self.old_policy_action_distributions)
        critic_distributions = torch.cat(self.critic_distributions, dim=0)
        batch_actions = torch.tensor(self.batch_actions)
        batch_rewards = torch.tensor(self.batch_rewards)
        batch_new_states = torch.tensor(self.batch_new_states)
        terminal_state = torch.tensor(self.terminal_state)

        print(f"batch_states = {batch_states}")
        print(f"policy_act_dist = {policy_action_distributions}")
        #print(f"batch_states = {batch_states}")
        #print(f"batch_states = {batch_states}")
        #print(f"batch_states = {batch_states}")
        #print(f"batch_states = {batch_states}")

        states = torch.squeeze(batch_states.unsqueeze(0).detach().to(device))
        policy_act_dist = torch.squeeze(torch.stack(policy_action_distributions, dim=0)).detach().to(device)
        old_policy_act_dist = torch.squeeze(torch.stack(old_policy_action_distributions, dim=0)).detach().to(device)
        #critic_dist = torch.squeeze(torch.stack(critic_distributions, dim=0)).detach().to(device)
        #batch_acts = torch.squeeze(torch.stack(batch_actions, dim=0)).detach().to(device)
        rewards = torch.squeeze(torch.stack(batch_rewards, dim=0)).detach().to(device)
        new_states = torch.squeeze(torch.stack(batch_new_states, dim=0)).detach().to(device)
        term_states = torch.squeeze(torch.stack(terminal_state, dim=0)).detach().to(device)
        
        # Compute advantage
        advantage = self.compute_advantage_function(states, rewards, new_states)

        # Compute probability ratio tensor. 
        prob_ratio = policy_act_dist/old_policy_act_dist
        prob_ratio = self.clip_ratio(prob_ratio)

        # compute the clipped_loss
        clipped_loss = prob_ratio*advantage

        # compute expected value of clipped loss
        loss = torch.mean(clipped_loss)

        # self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # clear all our memory
        self.batch_states = []
        self.policy_action_distributions = []
        self.old_policy_action_distributions = []
        self.critic_distributions = []
        self.batch_actions = []
        self.batch_rewards = []
        self.batch_new_states = []
        self.terminal_state = []
        
    def make_action(self, state):
        ''' This is different from the actor-critic networks action as we need to store the values we 
            obtain from this'''
        with torch.no_grad():
            # Run network to get nessessary variables, while ensuring pytorch does not compute gradients.
            state = torch.FloatTensor(state).to(device)
            action, log_action_distribution, critic_distribution, entropy = self.policy.forward(state)
            old_action, log_old_actor_distribution, old_critic_distribution, old_entropy = self.old_policy.forward(state)

        # Store the nessessary variables.
        print(np.array(log_action_distribution))
        self.batch_states.append(np.array(state))
        self.policy_action_distributions.append(log_action_distribution)
        self.old_policy_action_distributions.append(log_old_actor_distribution)
        self.critic_distributions.append(critic_distribution)
        self.batch_actions.append(action)


        return action

    def clip_ratio(self, prob_ratio):
        clipped_ratio = torch.clamp(prob_ratio, 1 - CLIP, 1 + CLIP)
        return torch.min(prob_ratio, clipped_ratio)

    def compute_advantage_function(self, states, rewards, new_states):
        '''
        params:
            states: the tensor of self.batch_states
            reward: the tensor of self.batch_rewards
            new_states: the tensor of self.batch_new_states

        This is the calculation of the advantage function.

        We can derive the advantage function in the paper to be equivilant to:
        A_t = sum_{t=0}^T(y^t r_t) + sum_{t=0}^(T-1)(y^{t+1}V_(t+1) - y^tV_t) 

        Using this, we can do the following:
        '''
        # best_action, _ = torch.max(self.critic_distributions, dim=0)
        indices = torch.arange(16).reshape(-1, 1)
        weights = torch.pow(GAMMA, indices)

        # we now calculated our weighted states and rewards. 
        weighted_rewards = rewards * weights
        weighted_states = states * weights
        weighted_new_states = new_states * weights

        # calculate advantage tensor
        advantage = torch.sum(weighted_rewards) + torch.sum(weighted_states - weighted_new_states, dim=1, keepdim=True)

        return advantage

    def next_state_append(self, reward, new_state, term_state):
        self.batch_rewards.append(reward)
        self.batch_new_states.append(new_state)
        self.terminal_state.append(term_state)
        
    def save(self, chkpt):
        torch.save(self.policy_old.state_dict(), chkpt)

    def load(self, chkpt):
        self.policy_old.load_state_dict(torch.load(chkpt, map_location=lambda storage, loc: storage))
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

if __name__ == '__main__':
    agent = PPO(365, 4)
    random_1 = torch.rand(365)
    for i in range(17):
        print(agent.make_action(random_1))

    agent.learn()