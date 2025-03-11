import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
import torch.optim as optim
from collections import deque
import torch.nn.functional as F
import os


class PPOMemory:
    def __init__(self,T,max_memory,batch_size,device):
        self.advantages = []
        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_logprobs = []
        self.buffer_state_values = []
        self.buffer_rewards = []
        self.buffer_is_terminal = []
        self.buffer_next_states = []

    def generate_batches():
        pass

    def remember(self,state,action,reward,next_state,done,log_probs,state_value,advantages):
        self.buffer_states.append(state)
        self.buffer_actions.append(action)
        self.buffer_rewards.append(reward)
        self.buffer_is_terminal.append(done)
        self.buffer_next_states.append(next_state)

        self.buffer_logprobs.append(log_probs)
        self.buffer_state_values.append(state_value)
        self.advantages.append(advantages)

    def clear_memory(self):
        self.memory = deque(maxlen=MAX_MEMORY)
    
class ActorNetwork(nn.Module):
    def __init__(self,input_size,hidden_size1,hidden_size2,hidden_size3,output_size,lr):
        super().__init__()
        self.linear1 = nn.Linear(input_size,hidden_size1)
        self.linear2 = nn.Linear(hidden_size1,hidden_size2)
        self.linear3 = nn.Linear(hidden_size2,hidden_size3)
        self.linear4 = nn.Linear(hidden_size3,output_size)
        self.optimizer = optim.Adam(self.actor_nework.parameters(),lr=lr)
        self.actor = nn.Sequential(
            self.linear1,
            nn.ReLU(),
            self.linear2,
            nn.ReLU(),
            self.linear3,
            nn.Softmax(dim=-1)
        )
        
    def forward(self,state):
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist
    
    def save(self,file_name="model1.pth"):
        model_folder_path = "./models"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path,file_name)
        torch.save(self.state_dict(),file_name)

class CriticalNetwork(nn.Module):
    def __init__(self,input_size,hidden_size1,hidden_size2,output_size,lr):
        super().__init__()
        self.linear1 = nn.Linear(input_size,hidden_size1)
        self.linear2 = nn.Linear(hidden_size1,hidden_size2)
        self.linear3 = nn.Linear(hidden_size2,output_size)
        self.optimizer = optim.Adam(self.actor_nework.parameters(),lr=lr)
        self.critic = nn.Sequential(
            self.linear1,
            nn.ReLU(),
            self.linear2,
            nn.ReLU(),
            self.linear3,
        )
        
    def forward(self,state):
        value = self.critic(state)
        return value

    def save(self,file_name="model1.pth"):
        model_folder_path = "./models"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path,file_name)
        torch.save(self.state_dict(),file_name)

class Agent:
    def __init__(self,gamma = 0.9,policy_clip = 0.3,batch_size = 64,N=2048,epochs = 10):
        self.actor_nework = ActorNetwork(
            input_size=10,
            hidden_size1=10,
            hidden_size2=10,
            hidden_size3=10,
            output_size=10,
            lr=10)
        self.critical_network = CriticalNetwork(
            input_size=10,
            hidden_size1=10,
            hidden_size2=10,
            output_size=10,
            lr=10)
        self.memory = PPOMemory(
            max_memory=10,
            batch_size=10,
        )
        self.gamma = gamma
        self.clip = policy_clip
        self.batch_size = batch_size
        self.N = N
        self.n_epochs = epochs #Numero de epocas que serão realizadas no processo de aprendizagem retirando informações da memória


    def remember(self,state,action,reward,next_state,done):
        self.memory.remember(state,action,reward,next_state,done)

    def save_models(self):
        print("... Saving models ...")
        self.actor_nework.save()
        self.critical_network.save()

    def load_models(self):
        print("... Saving models ...")
        self.actor_nework.load()
        self.critical_network.load()

    def choose_action(self,observation:np.array):
        state = torch.tensor([observation],dtype=torch.float)

        dist = self.actor_nework(state)
        value = self.critical_network(state)
        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return probs, action, value
    

    def generate_advantage_vector(self,rewards,states,next_states,state_values):
        for t in range(len(rewards)-1):
            discount = 1 #Valor de desconto que acumula os valores de gamma
            a_t = 0 #Incialização de valor de at neste time step
            delta_t = self.calculate_delta_t(t,rewards,state_values) #Calcular o valor do delta para o time step t

    def calculate_delta_t(self,t,rewards,state_values):
        return rewards[t]+self.gamma*state_values[t+1]-state_values[t]



    def get_L_clipped(self,r_theta,advatage):
        clipped = torch.clamp(prob_ratio,1-self.policy_clip,1+self.policy_clip)*advantage
        not_clipped = prob_ration*advantage
        return torch.min(clipped,not_clipped)



    def learn(self):
        for epoch in range(self.n_epochs):
            states,actions,old_probs,vals,rewards,dones,batches = self.memory.generate_batches()

            values = vals
            advantage = np.zeros(len(rewards),dtype=torch.float32)

            for t in range(len(rewards)-1):
                discount = 1
                a_t = 0
                for k in range(t,len(rewards)-1):
                    a_t += discount*(rewards[k]+self.gamma*values[k+1]*(1-int(dones[k]))-values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage)

            values = torch.tensor(values)
            for batch in batches:
                states = torch.tensor(states[batch],dtype=torch.float)
                old_probs = torch.tensor(old_probs[batch])
                actions = torch.tensor(actions[batch])

                dist = self.actor(states)
                critic_value = self.critical_network(states)

                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp()/old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio,1-self.policy_clip,1+self.policy_clip)*advantage[batch]
                actor_loss = -torch.min(weighted_probs,weighted_clipped_probs).mean()
                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor_nework.optimizer.zero_grad()
                self.critical_nework.optimizer.zero_grad()

                total_loss.backward()
                self.actor_nework.optimizer.step()
                self.critical_nework.optimizer.step()

        self.memory.clear_memory()

                


