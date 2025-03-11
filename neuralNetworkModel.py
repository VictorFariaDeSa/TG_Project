import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
import os
import torch.optim as optim

class DeepQNetwork(nn.Module):
    def __init__(self,lr:float,input_dims:int,fc1_dims:int,fc2_dims:int,fc3_dims:int,n_actions:int):
        super(DeepQNetwork,self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(in_features=input_dims,out_features=self.fc1_dims)
        self.fc2 = nn.Linear(in_features=self.fc1_dims,out_features=self.fc2_dims)
        self.fc3 = nn.Linear(in_features=self.fc2_dims,out_features=self.fc3_dims)
        self.fc4 = nn.Linear(in_features=self.fc3_dims,out_features=self.n_actions)
        self.optimizer = optim.Adam(self.parameters(),lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        self.to(self.device)

    
    def forward(self,observation):
        shape = observation.size()
        if (len(shape) == 1 and shape[0] != self.input_dims) or (len(shape) > 1 and shape[1] != self.input_dims):
            raise ValueError(
        f"O tensor passado como observação, de dimensões {list(shape)} possui as dimensões erradas. "
        f"Deve-se passar um tensor com {self.input_dims} colunas.")
        x=F.relu(self.fc1(observation))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        actions = self.fc4(x)
        return actions
    
class Agent():
    def __init__(self, gamma:float, 
                 epsilon:float, 
                 lr:float, 
                 input_dims:int, 
                 batch_size:int, 
                 n_actions:int, 
                 max_mem_size:int = 100000, 
                 eps_end:float=0.01, 
                 eps_dec:float=2e-7,
                 num_actions_p_joint:int = 3):
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.num_actions_p_joint = num_actions_p_joint
        self.n_actions = n_actions
        self.n_joints = int(self.n_actions/self.num_actions_p_joint)

        self.mem_counter:int = 0

        self.Q_eval:DeepQNetwork = DeepQNetwork(lr = self.lr,
                                                n_actions=n_actions,
                                                input_dims=input_dims,
                                                fc1_dims=512,
                                                fc2_dims=256,
                                                fc3_dims=256,)

        self.state_memory = np.zeros((self.mem_size,input_dims),dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size,input_dims),dtype=np.float32)

        self.action_memory = np.zeros((self.mem_size,self.n_joints),dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size,dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size,dtype=bool)

    def store_transitions(self,state,action,reward,state_,done):
        index = self.mem_counter%self.mem_size
        self.state_memory[index] = state.cpu().numpy()
        self.action_memory[index] = action.cpu().numpy()
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_.cpu().numpy()    
        self.terminal_memory[index] = done

        self.mem_counter+=1

    def policy(self,observation):
        nn_prediction = self.Q_eval(observation)
        return self.choose_action(nn_prediction)

    def choose_action(self, nn_prediction, e=None):
        if e is None:
            e = self.epsilon
        elif e > 1 or e < 0:
            raise ValueError("O parametro e inserido é invalido")
        
        final_actions = []
        
        if nn_prediction.ndimension() == 1:
            nn_prediction = nn_prediction.unsqueeze(0)
        for row in nn_prediction:
            row_actions = []
            for i in range(0, self.n_actions, self.num_actions_p_joint):
                random_number = np.random.random()
                if random_number > e:
                    group = row[i:i+self.num_actions_p_joint]
                    action_index = i + torch.argmax(group).item()
                else:
                    action_index = np.random.choice(np.arange(i, i + self.num_actions_p_joint))  # Escolhe uma ação aleatória
                
                row_actions.append(action_index)
            
            final_actions.append(row_actions)
        
        return torch.tensor(final_actions).squeeze()
    
    def learn(self):
        if self.mem_counter < self.batch_size:
            print("A memória do processo ainda esta muito pequena, aguardando mais resultados de simulação")
            return
        self.Q_eval.optimizer.zero_grad()
        max_mem = min(self.mem_counter, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        state_batch = torch.tensor(self.state_memory[batch], device=self.Q_eval.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch], device=self.Q_eval.device)
        reward_batch = torch.tensor(self.reward_memory[batch], device=self.Q_eval.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch], device=self.Q_eval.device)
        action_batch = torch.tensor(self.action_memory[batch], dtype=torch.long, device=self.Q_eval.device)

        
        q_eval = self.Q_eval.forward(state_batch)
        q_real = torch.clone(q_eval)

        
        q_next = self.Q_eval.forward(new_state_batch)


        linha_zeros = torch.zeros((1, self.n_actions))
        q_next = q_next.float()
        q_next[terminal_batch.flatten()] = linha_zeros.to(self.Q_eval.device)
        
        q_next_actions = self.choose_action(q_next,e=0)
        q_next_mean_reward = torch.mean(torch.gather(q_next, 1, q_next_actions.to(q_next.device)), dim=1)


        q_target = (reward_batch +self.gamma * q_next_mean_reward).unsqueeze(1)
        rows = torch.arange(self.batch_size).unsqueeze(1)
        q_real[rows, action_batch] = q_target


        loss = self.Q_eval.loss(q_real,q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
    
    def LoadModel(self,path):
        if os.path.exists(path):
            print("Arquivo encontrado. Carregando o modelo...")
            self.Q_eval.load_state_dict(torch.load(path))
            self.Q_eval.train()
        else:
            print("Arquivo não encontrado. Treinando e salvando o modelo...")
    
    def saveModel(self,model_path):
        torch.save(self.Q_eval.state_dict(), model_path)
        print("Modelo salvo com sucesso!")

class RobotNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_hidden_layer = nn.Linear(in_features=21, out_features=100)
        # self.second_hidden_layer = nn.Linear(in_features=255, out_features=255)
        self.third_hidden_layer = nn.Linear(in_features=100, out_features=24)

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.first_hidden_layer(x))
        # x = F.relu(self.second_hidden_layer(x))
        x = self.third_hidden_layer(x)
        return x
    
class Policy():

    def e_greedy_choose(self,q_values,e):
        if e > 1 or e < 0:
            raise ValueError("O parametro e inserido é invalido")
        random_number = random.random()
        if random_number <= e:
            choice = random.choice(range(len(q_values)))
            if choice > 2:
                raise ValueError("A escolha da politica foge ao escopo")
            return choice
        else:
            return np.argmax(q_values)

def loadModel(model_path):

    nn_model = RobotNeuralNetwork()

    if os.path.exists(model_path):
        print("Arquivo encontrado. Carregando o modelo...")
        nn_model.load_state_dict(torch.load(model_path))
        nn_model.train()
    else:
        print("Arquivo não encontrado. Treinando e salvando o modelo...")
    return nn_model

def saveModel(model,model_path):
    torch.save(model.state_dict(), model_path)
    print("Modelo salvo com sucesso!")