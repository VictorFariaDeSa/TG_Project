import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self,input_size,hidden_size1,hidden_size2,hidden_size3,output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size,hidden_size1)
        self.linear2 = nn.Linear(hidden_size1,hidden_size2)
        self.linear3 = nn.Linear(hidden_size2,hidden_size3)
        self.linear4 = nn.Linear(hidden_size3,output_size)

        
    def forward(self,x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x
    
    def save(self,file_name="model1.pth"):
        model_folder_path = "./models"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path,file_name)
        torch.save(self.state_dict(),file_name)

class QTrainer:
    def __init__(self,model,lr,gamma):
        self.lr = lr
        self.model = model
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(),lr = self.lr)
        self.criterion = nn.MSELoss()

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

        pred = self.model(state)

        target = pred.clone()
        for index in range(len(done)):
            q_new = reward[index]
            if not done[index]:
                q_new = reward[index] +self.gamma * torch.max(self.model(new_state[index]))
            target[index][torch.argmax(action).item()] = q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target,pred)
        loss.backward()

        self.optimizer.step()