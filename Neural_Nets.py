from torch import nn
import torch
import torch.optim as optim
from torch.distributions import Normal
import os



class PPO_NN(nn.Module):
    def __init__(self, num_inputs, num_outputs, inner_dimensions, std=0.0):
        super(PPO_NN, self).__init__()

        critic_layers = []
        actor_layers = []
        in_features = num_inputs

        for out_features in inner_dimensions:
            critic_layers.append(nn.Linear(in_features, out_features))
            critic_layers.append(nn.ReLU())
            actor_layers.append(nn.Linear(in_features, out_features))
            actor_layers.append(nn.ReLU())
            in_features = out_features
        
        critic_layers.append(nn.Linear(in_features, 1))
        actor_layers.append(nn.Linear(in_features,num_outputs))


        self.critic = nn.Sequential(*critic_layers)
        self.actor = nn.Sequential(*actor_layers)
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

    def forward(self, x):
        value = self.critic(x)
        mu    = torch.tanh(self.actor(x))
        std = self.log_std.squeeze(0).exp().expand_as(mu)
        dist  = Normal(mu, std)
        return dist, value
    
    def save(self,actor_file_name="actor_model.pth",critic_file_name = "critic_model.pth"):
        model_folder_path = "./models"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        complete_actor_name = os.path.join(model_folder_path,actor_file_name)
        complete_critic_name = os.path.join(model_folder_path,critic_file_name)  
        torch.save(self.critic.state_dict(),complete_critic_name)
        torch.save(self.actor.state_dict(),complete_actor_name)

class ActorNetwork(nn.Module):
    def __init__(self,input_size,hidden_size1,hidden_size2,hidden_size3,output_size,lr):
        super().__init__()
        self.linear1 = nn.Linear(input_size,hidden_size1)
        self.linear2 = nn.Linear(hidden_size1,hidden_size2)
        self.linear3 = nn.Linear(hidden_size2,hidden_size3)
        self.linear4 = nn.Linear(hidden_size3,output_size)
        self.optimizer = optim.Adam(self.parameters(),lr=lr)
        self.base = nn.Sequential(
            self.linear1,
            nn.Tanh(),
            self.linear2,
            nn.Tanh(),
            self.linear3,
            nn.Tanh(), 
        )
        self.mu_head = self.linear4
        self.log_std = nn.Parameter(torch.zeros(output_size))  # desvio padrão aprendido (fixo para todos os estados)

    def forward(self, state):
        x = self.base(state)
        mu = self.mu_head(x)
        std = torch.exp(self.log_std)  # garantir que std seja positivo
        dist = Normal(mu, std)
        return dist
    
    def save(self,file_name="actor_model.pth"):
        model_folder_path = "./models"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path,file_name)
        torch.save(self.state_dict(),file_name)

class CriticalNetwork(nn.Module):
    def __init__(self,input_size,hidden_size1,hidden_size2,hidden_size3,output_size,lr):
        super().__init__()
        self.linear1 = nn.Linear(input_size,hidden_size1)
        self.linear2 = nn.Linear(hidden_size1,hidden_size2)
        self.linear3 = nn.Linear(hidden_size2,hidden_size3)
        self.linear4 = nn.Linear(hidden_size3,output_size)
        self.optimizer = optim.Adam(self.parameters(),lr=lr)
        self.critic = nn.Sequential(
            self.linear1,
            nn.ReLU(),
            self.linear2,
            nn.ReLU(),
            self.linear3,
            nn.ReLU(),
            self.linear4
        )
        
    def forward(self,state):
        value = self.critic(state)
        return value

    def save(self,file_name="critic_model.pth"):
        model_folder_path = "./models"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path,file_name)
        torch.save(self.state_dict(),file_name)