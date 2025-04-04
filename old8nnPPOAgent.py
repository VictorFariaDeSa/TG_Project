import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import numpy as np
import os
from SimulationControl import createSimulation
from helper import plot

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
    np.array(self.buffer_probs),np.array(self.buffer_state_values),np.array(self.advantages),\
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
        self.advantages = []
        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_probs = []
        self.buffer_state_values = []
        self.buffer_rewards = []
        self.buffer_is_terminal = []
        self.buffer_next_states = []
        self.buffer_returns = []
    
class ActorNetwork(nn.Module):
    def __init__(self,input_size,hidden_size1,hidden_size2,hidden_size3,output_size,lr):
        super().__init__()
        self.linear1 = nn.Linear(input_size,hidden_size1)
        self.linear2 = nn.Linear(hidden_size1,hidden_size2)
        self.linear3 = nn.Linear(hidden_size2,hidden_size3)
        self.linear4 = nn.Linear(hidden_size3,output_size)
        self.optimizer = optim.Adam(self.parameters(),lr=lr)
        self.actor = nn.Sequential(
            self.linear1,
            nn.ReLU(),
            self.linear2,
            nn.ReLU(),
            self.linear3,
            nn.ReLU(),
            self.linear4,
            nn.Softmax(dim=-1)
        )
        
    def forward(self,state):
        probs = self.actor(state)
        return Categorical(probs)
    
    def save(self,file_name="model1.pth"):
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

    def save(self,file_name="model1.pth"):
        model_folder_path = "./models"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path,file_name)
        torch.save(self.state_dict(),file_name)


class Agent:
    def __init__(self,gamma = 0.99,policy_clip = 0.2,batch_size = 10,epochs = 5,n_joints = 8,input_size = 23,output_size = 3,N = 50):
        self.critical_network = CriticalNetwork(
            input_size = input_size,
            hidden_size1 = 512,
            hidden_size2 = 256,
            hidden_size3 = 256,
            output_size = 1,
            lr=1e-5)
        self.actor_networks = [ActorNetwork(input_size=input_size,
            hidden_size1 = 512,
            hidden_size2 = 256,
            hidden_size3 = 128,
            output_size = output_size,
            lr=1e-4) for _ in range(n_joints)]
        self.gamma = gamma
        self.clip_epsilon = policy_clip
        self.batch_size = batch_size
        self.memory = PPOMemory(batch_size=self.batch_size)
        self.n_epochs = epochs #Numero de epocas que serão realizadas no processo de aprendizagem retirando informações da memória
        self.n_games = 0
        self.n_steps = 0
        self.N = N

    def save_models(self):
        print("... Saving models ...")
        for i, actor in enumerate(self.actor_networks):
            actor.save(file_name=f"actor_model{i}.pth")
        self.critical_network.save(file_name="critical_model.pth")

    def load_models(self):
        for i,actor in enumerate(self.actor_networks):
            model_path = f"models/actor_model{i}.pth"
            if os.path.exists(model_path):
                print(f"... Loading {model_path} ...")
                actor.load_state_dict(torch.load(model_path))
            else:
                print(f"...Arquivo {model_path} não encontrado, o modelo será gerado do zero")
        model_path = "models/critical_model.pth"
        if os.path.exists(model_path):
            print(f"... Loading {model_path} ...")
            self.critical_network.load_state_dict(torch.load(model_path))
        else:
            print(f"...Arquivo {model_path} não encontrado, o modelo será gerado do zero")


    def get_state(self,env):
        return env.getObservation()

    def remember(self,state,actions,reward,next_state,done,log_probs,state_value):
        self.memory.remember(state,actions,reward,next_state,done,log_probs,state_value)

    def calculate_returns(self, rewards):
        returns = np.zeros_like(rewards, dtype=np.float32)
        future_reward = 0.0  

        for t in reversed(range(len(rewards))):
            future_reward = rewards[t] + self.gamma * future_reward
            returns[t] = future_reward

        self.memory.buffer_returns = returns.tolist()


    def calculate_advantages_GAE(self, lam=0.95):
        gamma = self.gamma
        rewards = self.memory.buffer_rewards
        values = self.memory.buffer_state_values
        advantages = np.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] - values[t] if t < len(rewards) - 1 else rewards[t] - values[t] #Calculo da vantagem específica do tempo T
            gae = delta + gamma * lam * gae
            advantages[t] = gae
        self.memory.advantages = advantages.tolist()

    def calculate_r_theta(self,state,old_probs,action,network):       
        dist = network(state.to(torch.float))
        new_probs = dist.log_prob(action)
        r_theta = torch.exp(new_probs - old_probs)
        
        
        
        # print(f"state:{state.shape}")
        # print(f"dist:{dist.probs.shape}")
        # print(f"new_probs:{new_probs.shape}")
        # print(f"r_theta: {(r_theta).shape}")
        # print("---"*30)
        # print(dist.logits)
        # print(action)
        # print(new_probs)
        # print(r_theta)
        # print("---"*30)

        return r_theta
    
    def get_L_clipped(self, r_theta, advantage, dist, c2=0.01):
        clipped = torch.clamp(r_theta, 1-self.clip_epsilon, 1+self.clip_epsilon)*advantage
        not_clipped = r_theta*advantage
        min_loss = -torch.min(clipped, not_clipped)
        
        # Add entropy bonus
        entropy = dist.entropy().mean()
        entropy_bonus = -c2 * entropy  # Negative because we want to maximize entropy
        
        return min_loss + entropy_bonus


    # def get_L_clipped(self,r_theta, advantage):
    #     clipped = torch.clamp(r_theta,1-self.clip_epsilon,1+self.clip_epsilon)*advantage
    #     not_clipped = r_theta*advantage
    #     min_loss = -torch.min(clipped,not_clipped)

    #     # print(r_theta)
    #     # print(advantage)
    #     # print(not_clipped)

    #     # print(f"clipped: {clipped.shape}")
    #     # print(f"not_clipped: {not_clipped.shape}")
    #     # print(f"min_loss: {min_loss.shape}")



    #     return min_loss


    def choose_actions(self,state):
        actions = []
        probs = []
        value = self.critical_network(state)
        value = torch.squeeze(value).item()
        # print("-"*50)
        for network in self.actor_networks:
            dist = network(state)
            # print(dist.probs)
            action = dist.sample()
            prob = torch.squeeze(dist.log_prob(action)).item()
            # print(dist.probs[action].item())
            # prob = torch.squeeze(dist.probs[action]).item()
            action = torch.squeeze(action).item()
            probs.append(prob)
            actions.append(action)

        return actions,probs,value

    def train_step_batch(self,states,probs,advantages,state_values,returns,actions,c1):
        policy_loss = 0
        for i,network in enumerate(self.actor_networks):
            curr_actions = actions[:,i]
            curr_probs = probs[:,i]
            dist = network(states.to(torch.float))
            r_theta = self.calculate_r_theta(state=states,old_probs=curr_probs,action=curr_actions,network=network)
            l_clip = self.get_L_clipped(r_theta,advantages,dist=dist).mean()

            policy_loss += l_clip

        critic_value = torch.squeeze(self.critical_network(states.to(torch.float32)))
        critic_loss = ((returns-critic_value)**2).mean()


        total_loss = policy_loss + c1*critic_loss
            
        for network in self.actor_networks:
            network.optimizer.zero_grad()
        

        policy_loss.backward()

        for network in self.actor_networks:
            network.optimizer.step()

        self.critical_network.optimizer.zero_grad()
        critic_loss.backward()
        self.critical_network.optimizer.step()

        return critic_loss


    def learn(self):
        self.calculate_returns(self.memory.buffer_rewards)
        for epoch in range(self.n_epochs):
            self.calculate_advantages_GAE(lam=0.95)
            states,actions,rewards,terminals,\
            next_states,probs,state_values,\
            advantages, returns, batches = self.memory.generate_batches()
            

            l_mse = []
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

                # nn_pred = self.critical_network(torch.tensor(t_next_states,dtype=torch.float)).squeeze()
                # returns = t_rewards+self.gamma*nn_pred
                #Calcular os retornos direto aqui não seria melhor?
                t_advantages = (t_advantages - t_advantages.mean()) / (t_advantages.std() + 1e-8)


                mse = self.train_step_batch(t_states,t_old_probs,t_advantages,t_state_values,t_returns,t_actions,c1 = 0.5)
                l_mse.append(mse)
        self.memory.clear_memory()
        return sum(l_mse) / len(l_mse)

            
def train():
    plot_scores = []
    plot_mean_scores = []
    plot_mse = []
    record = float("-inf")
    agent = Agent()
    agent.load_models()
    simulation = createSimulation("cpu")
    simulation.reset()
    last_scores = deque(maxlen=100)
    while True:
        state_old = agent.get_state(env = simulation)
        final_move,action_probs,state_value = agent.choose_actions(state = torch.tensor(state_old,dtype=torch.float))
        reward, done, score = simulation.step(actions = final_move)
        state_new = agent.get_state(env = simulation)
        agent.remember(state=state_old,actions=final_move,reward=reward,
                       next_state=state_new,done=done,state_value=state_value,log_probs=action_probs)
        if done:
            print("done")
            temp_mse = agent.learn()
            simulation.reset()
            agent.n_games += 1
            plot_mse.append(temp_mse.item())

            if score > record:
                record = score
                agent.save_models()

            print(f"Game:{agent.n_games}   |Score:{score}     |Record:{record}")
            plot_scores.append(score)
            last_scores.append(score)
            mean_score = sum(last_scores)/len(last_scores)
            plot_mean_scores.append(mean_score)
            plot(plot_scores,plot_mean_scores,plot_mse)


if __name__ == "__main__":
    train()