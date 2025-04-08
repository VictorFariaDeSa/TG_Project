import torch
import torch.nn as nn
from torch.distributions import Normal
import argparse
import math
import os
import random
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from SimulationControl import createSimulation
from helper import plot
from collections import deque


LEARNING_RATE       = 1e-4
GAMMA               = 0.99
GAE_LAMBDA          = 0.95
POLICY_CLIP         = 0.2
CRITIC_DISCOUNT     = 0.5
ENTROPY_BETA        = 0.001
PPO_STEPS           = 1000
MINI_BATCH_SIZE     = 64
N_EPOCHS            = 10


NUM_INPUTS = 23
NUM_OUTPUTS = 8
HIDDEN_SIZE1 = 512
HIDDEN_SIZE2 = 216
HIDDEN_SIZE3 = 128


class PPOMemory:
    def __init__(self,batch_size):
        self.batch_size = batch_size
        self.clear_memory()

    def generate_batches(self):
        n_states = len(self.buffer_states)
        batch_start = np.arange(0,n_states,self.batch_size)
        indices = np.arange(n_states)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        return np.array(self.buffer_states),np.array(self.buffer_actions),\
    np.array(self.buffer_rewards),np.array(self.buffer_is_terminal),np.array(self.buffer_next_states),\
    np.array(self.buffer_probs),np.array(self.buffer_state_values),np.array(self.buffer_advantages),\
    np.array(self.buffer_returns),batches

    def remember(self,state,actions,reward,next_state,done,log_probs,state_value):
        self.buffer_states.append(state)
        self.buffer_actions.append(actions)
        self.buffer_rewards.append(reward)
        self.buffer_is_terminal.append(done)
        self.buffer_next_states.append(next_state)

        self.buffer_probs.append(log_probs)
        self.buffer_state_values.append(state_value)

    def clear_memory(self):
        self.buffer_advantages = []
        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_probs = []
        self.buffer_state_values = []
        self.buffer_rewards = []
        self.buffer_is_terminal = []
        self.buffer_next_states = []
        self.buffer_returns = []

class PPO_NN(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size1,hidden_size2,hidden_size3, std=0.0):
        super(PPO_NN, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, hidden_size3),
            nn.ReLU(),
            nn.Linear(hidden_size3, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, hidden_size3),
            nn.ReLU(),
            nn.Linear(hidden_size3, num_outputs)
        )

        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        
    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x).unsqueeze(0)
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        return dist, value

class PPO_Agent:
    def __init__(self):
        self.PPO_network = PPO_NN(num_inputs=NUM_INPUTS,num_outputs=NUM_OUTPUTS,hidden_size1=HIDDEN_SIZE1,hidden_size2=HIDDEN_SIZE2,hidden_size3=HIDDEN_SIZE3)
        self.optimizer = optim.Adam(self.PPO_network.parameters(), lr=LEARNING_RATE)
        self.gamma = GAMMA
        self.clip_epsilon = POLICY_CLIP
        self.batch_size = MINI_BATCH_SIZE
        self.memory = PPOMemory(batch_size=self.batch_size)
        self.n_epochs = N_EPOCHS
        self.max_steps = PPO_STEPS
        self.lam = GAE_LAMBDA

        self.n_games = 0
        self.n_steps = 0

    def compute_gae(self,next_value):
        values = self.memory.buffer_state_values + [next_value]
        gae = 0
        advantages = np.zeros_like(self.memory.buffer_rewards)
        for step in reversed(range(len(self.memory.buffer_rewards))):
            delta = self.memory.buffer_rewards[step] + self.gamma * values[step + 1] * (1-self.memory.buffer_is_terminal[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1-self.memory.buffer_is_terminal[step]) * gae
            advantages[step] = gae
        self.memory.buffer_returns = advantages.tolist()


    def train_step_batch(self,states,probs,advantages,state_values,returns,actions):
        action = actions
        old_log_probs = probs
        dist, value = self.PPO_network(states.to(torch.float))
        entropy = dist.entropy().mean()
        new_log_probs = dist.log_prob(action).squeeze()
        ratio = (new_log_probs - old_log_probs).exp()

        adv = advantages.unsqueeze(1)       
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv
        actor_loss  = -torch.min(surr1, surr2).mean()
        critic_loss = (returns - value).pow(2).mean()
        loss = CRITIC_DISCOUNT * critic_loss + actor_loss - ENTROPY_BETA * entropy   
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss



    def ppo_update(self):
        l_mse = []
        # PPO EPOCHS is the number of times we will go through ALL the training data to make updates
        for _ in range(N_EPOCHS):
            states,actions,rewards,terminals,\
            next_states,probs,state_values,\
            advantages, returns, batches = self.memory.generate_batches()

            for batch in batches:
                if batch.size == 1:
                    continue
                t_actions = torch.tensor(actions[batch])
                t_states = torch.tensor(states[batch])
                t_next_states = torch.tensor(next_states[batch])
                t_old_probs = torch.tensor(probs[batch])
                t_state_values = torch.tensor(state_values[batch])
                t_advantages = torch.tensor(advantages[batch])
                t_rewards = torch.tensor(rewards[batch])
                t_returns = torch.tensor(returns[batch])

            mse = self.train_step_batch(states=t_states,probs=t_old_probs,advantages=t_advantages,state_values=t_state_values,returns=t_returns,actions=t_actions)
            l_mse.append(mse)
        self.memory.clear_memory()
        return sum(l_mse) / len(l_mse)




def normalize(x):
    x -= x.mean()
    x /= (x.std() + 1e-8)
    return x


if __name__ == "__main__":
    
    plot_scores = []
    plot_mean_scores = []
    plot_mse = []
    record = float("-inf")
    last_scores = deque(maxlen=100)


    env = createSimulation("cpu")
    env.reset()
    agent = PPO_Agent()

    while True:
        for _ in range(agent.max_steps):
            old_state = env.getObservation()
            dist, value = agent.PPO_network(torch.tensor(old_state,dtype=torch.float))
            value = value.item()
            action = dist.sample().squeeze(0)
            reward, done, score = env.C_step(action)
            new_state = env.getObservation()
            log_prob = dist.log_prob(action).squeeze()
            
            agent.memory.remember(state=old_state,actions=action,reward=reward,done=done,log_probs=log_prob.detach().numpy(),next_state=new_state,state_value=value)

            if done:
                break

        _,next_value = agent.PPO_network(torch.tensor(new_state,dtype=torch.float))
        agent.compute_gae(next_value)
        advantage = torch.tensor(agent.memory.buffer_returns) - torch.tensor(agent.memory.buffer_state_values)
        agent.memory.buffer_advantages = normalize(advantage)
        
        temp_mse = agent.ppo_update()
        plot_mse.append(temp_mse.item())
        env.reset()
        agent.n_games+=1
        if score > record:
                record = score
                # agent.save_models()

        print(f"Game:{agent.n_games}   |Score:{score}     |Record:{record}")
        plot_scores.append(score)
        last_scores.append(score)
        mean_score = sum(last_scores)/len(last_scores)
        plot_mean_scores.append(mean_score)
        plot(plot_scores,plot_mean_scores,plot_mse)