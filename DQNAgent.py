import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.optim as optim
import torch.nn.functional as F

import random
from collections import deque
import numpy as np
import os
from SimulationControl import createSimulation
from helper import plot

MAX_MEMORY = 10000

class DQNMemory:
    def __init__(self,batch_size):
        self.clear_memory()

    def generate_batches(self):
        n_states = len(self.buffer_states)
        batch_start = np.arange(0,n_states,self.batch_size)
        indices = np.arange(n_states)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        return np.array(self.buffer_states),np.array(self.buffer_actions),\
    np.array(self.buffer_rewards),np.array(self.buffer_is_terminal),np.array(self.buffer_next_states),\
    np.array(self.buffer_probs),np.array(self.buffer_state_values),np.array(self.advantages),\
    np.array(self.buffer_returns),batches

    def remember(self,state,actions,reward,next_state,done):
        self.buffer_states.append(state)
        self.buffer_actions.append(actions)
        self.buffer_rewards.append(reward)
        self.buffer_next_states.append(next_state)
        self.buffer_is_terminal.append(done)

    def clear_memory(self,maxlen = MAX_MEMORY):
        self.buffer_states = deque(maxlen=maxlen)
        self.buffer_actions = deque(maxlen=maxlen)
        self.buffer_rewards = deque(maxlen=maxlen)
        self.buffer_next_states = deque(maxlen=maxlen)
        self.buffer_is_terminal = deque(maxlen=maxlen)



class Linear_QNet(nn.Module):
    def __init__(self,input_size,hidden_size1,hidden_size2,hidden_size3,output_size,lr):
        super().__init__()
        self.linear1 = nn.Linear(input_size,hidden_size1)
        self.linear2 = nn.Linear(hidden_size1,hidden_size2)
        self.linear3 = nn.Linear(hidden_size2,hidden_size3)
        self.linear4 = nn.Linear(hidden_size3,output_size)
        self.lr = lr
        self.optimizer = optim.Adam(self.parameters(),lr = self.lr)
        self.criterion = nn.MSELoss()
        self.q_net = nn.Sequential(
            self.linear1,
            nn.ReLU(),
            self.linear2,
            nn.ReLU(),
            self.linear3,
            nn.ReLU(),
            self.linear4
        )
        
    def forward(self,state):
        value = self.q_net(state)
        return value
    
    def save(self,file_name="model1.pth"):
        model_folder_path = "./models"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path,file_name)
        torch.save(self.state_dict(),file_name)


class Agent:

    def __init__(self,gamma = 0.9, epsilon = 0.2, batch_size = 64, epochs = 10):
        self.q_net = Linear_QNet(24,256,128,64,24,0.001)
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.memory = DQNMemory(batch_size=self.batch_size)
        self.n_epochs = epochs #Numero de epocas que serão realizadas no processo de aprendizagem retirando informações da memória
        self.n_games = 0
        

    def save_models(self):
        print("... Saving models ...")
        self.q_net.save()

    def load_models(self):
        print("... Loading models ...")
        self.q_net.load()

    def get_state(self,env):
        return env.getObservation()

    def remember(self,state,actions,reward,next_state,done):
        self.memory.remember(state,actions,reward,next_state,done)

    def train_step(self,state,action,reward,new_state,done):
        
        state = torch.tensor(state,dtype=torch.float)
        action = torch.tensor(action,dtype=torch.long)
        reward = torch.tensor(reward,dtype=torch.float)
        new_state = torch.tensor(new_state,dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state,0)
            action = torch.unsqueeze(action,0)
            new_state = torch.unsqueeze(new_state,0)
            reward = torch.unsqueeze(reward,0)
            done = (done,)

        pred = self.q_net(state)

        target = pred.clone()
        for index in range(len(done)):
            q_new = reward[index]
            if not done[index]:
                q_new = reward[index] +self.gamma * torch.max(self.q_net(new_state[index]))
            target[index][torch.argmax(action).item()] = q_new
        self.q_net.optimizer.zero_grad()
        loss = self.q_net.criterion(target,pred)
        loss.backward()

        self.q_net.optimizer.step()



    def train_long_memory(self):
        if len(self.memory.buffer_rewards) >= self.batch_size:
            states = random.sample(self.memory.buffer_states,self.batch_size)
            actions = random.sample(self.memory.buffer_actions,self.batch_size)
            rewards = random.sample(self.memory.buffer_rewards,self.batch_size)
            next_states = random.sample(self.memory.buffer_next_states,self.batch_size)
            dones = random.sample(self.memory.buffer_is_terminal,self.batch_size)
        else:
            states = self.memory.buffer_states
            actions = self.memory.buffer_rewards
            rewards = self.memory.buffer_actions
            next_states = self.memory.buffer_next_states
            dones = self.memory.buffer_is_terminal

        self.train_step(states,actions,rewards,next_states,dones)
        

    def train_short_memory(self,state,action,reward,next_state,done):
        self.train_step(state,action,reward,next_state,done)


    def choose_action(self,state):
        final_move = []
        prediction = self.q_net(torch.tensor(state,dtype=torch.float))
        prediction_reshaped = prediction.view(-1, 3)
        for joint in prediction_reshaped:
            if np.random.random() < self.epsilon:
                final_move.append(random.randint(0,2))
            else:
                final_move.append(torch.argmax(joint))
        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    plot_mse = []
    total_score = 0
    record = float("-inf")
    agent = Agent()
    simulation = createSimulation("cpu")
    simulation.reset()
    while True:
        state_old = agent.get_state(simulation)
        final_move = agent.choose_action(state_old)
        reward, done, score = simulation.step(final_move)
        state_new = agent.get_state(simulation)
        agent.train_short_memory(state=state_old,action=final_move,reward=reward,next_state=state_new,done=done)
        agent.remember(state_old,final_move,reward,state_new,done)


        if done:
            agent.train_long_memory()
            simulation.reset()
            agent.n_games += 1

            if score > record:
                record = score
                # agent.model.save()
            print(f"Game:{agent.n_games}   |Score:{score}     |Record:{record}")
            plot_scores.append(score)
            total_score += score
            mean_score = total_score/agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores,plot_mean_scores,None)

if __name__ == "__main__":
    train()


