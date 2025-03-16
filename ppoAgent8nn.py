import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
import torch.optim as optim
import torch.nn.functional as F
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
    def __init__(self,gamma = 0.9,policy_clip = 0.3,batch_size = 64,N=2048,epochs = 10,n_joints = 8,input_size = 21,output_size = 3):
        self.critical_network = CriticalNetwork(
            input_size = 21,
            hidden_size1 = 512,
            hidden_size2 = 256,
            hidden_size3 = 256,
            output_size = 1,
            lr=1e-4)
        self.actor_networks = [ActorNetwork(input_size=input_size,
            hidden_size1 = 512,
            hidden_size2 = 256,
            hidden_size3 = 256,
            output_size = output_size,
            lr=1e-6) for _ in range(n_joints)]
        self.gamma = gamma
        self.clip_epsilon = policy_clip
        self.batch_size = batch_size
        self.memory = PPOMemory(batch_size=self.batch_size)
        self.N = N
        self.n_epochs = epochs #Numero de epocas que serão realizadas no processo de aprendizagem retirando informações da memória
        self.n_games = 0

    def save_models(self):
        print("... Saving models ...")
        self.actor_network.save()
        self.critical_network.save()

    def load_models(self):
        print("... Loading models ...")
        self.actor_network.load()
        self.critical_network.load()

    def get_state(self,env):
        return env.getObservation()

    def remember(self,state,actions,reward,next_state,done,log_probs,state_value):
        self.memory.remember(state,actions,reward,next_state,done,log_probs,state_value)

    def calulate_returns(self,rewards):
        for t in range(len(rewards)):
            discout = 1
            discouted_reward = 0
            for next_t in range(t,len(rewards)):
                discouted_reward += rewards[next_t]*discout
                discout *= self.gamma
            self.memory.buffer_returns.append(discouted_reward)
    


    def calculate_advantages_GAE(self, gamma=0.99, lam=0.95):
        rewards = self.memory.buffer_rewards
        values = self.memory.buffer_state_values
        advantages = np.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] - values[t] if t < len(rewards) - 1 else rewards[t] - values[t] #Calculo da vatagem específica do tempo T
            gae = delta + gamma * lam * gae
            advantages[t] = gae + values[t]
        self.memory.advantages = advantages.tolist()

    def calculate_r_theta(self,state,old_probs,action,network):       
        dist = network(state.to(torch.float))
        new_probs = dist.log_prob(action)
        # new_probs = dist.probs.gather(1,action.unsqueeze(1)).squeeze()
        return new_probs.exp()/old_probs.exp()
        
    def get_L_clipped(self,r_theta, advantage):
        clipped = torch.clamp(r_theta,1-self.clip_epsilon,1+self.clip_epsilon)*advantage.view(-1, 1)
        not_clipped = r_theta*advantage.view(-1, 1)
        return -torch.min(clipped,not_clipped)
    
    def complete_vectors(self):
        self.calulate_returns(rewards=self.memory.buffer_rewards)
        self.calculate_advantages_GAE(gamma=self.gamma,lam=0.95)

    def choose_actions(self,state):
        actions = []
        probs = []
        value = self.critical_network(state)
        value = torch.squeeze(value).item()
        for network in self.actor_networks:
            dist = network(state)
            action = dist.sample()
            prob = torch.squeeze(dist.log_prob(action)).item()
            # print(dist.probs[action].item())
            # prob = torch.squeeze(dist.probs[action]).item()
            action = torch.squeeze(action).item()
            probs.append(prob)
            actions.append(action)

        return actions,probs,value

    def train_step_batch(self,states,probs,advantages,state_values,returns,actions):
        for i,network in enumerate(self.actor_networks):
            curr_actions = actions[:,i]
            curr_probs = probs[:,i]

            r_theta = self.calculate_r_theta(state=states,old_probs=curr_probs,action=curr_actions,network=network)
            l_clip = self.get_L_clipped(r_theta,advantages).mean()

            network.optimizer.zero_grad() 
            l_clip.backward()
            network.optimizer.step()

        critic_value = torch.squeeze(self.critical_network(states.to(torch.float32)))

        
        critic_loss = F.mse_loss(critic_value, returns.to(torch.float32))
        self.critical_network.optimizer.zero_grad()
        critic_loss.backward()
        self.critical_network.optimizer.step()
        return critic_loss

    def learn(self):
        for epoch in range(self.n_epochs):
            states,actions,rewards,terminals,\
            next_states,probs,state_values,\
            advantages, returns, batches = self.memory.generate_batches()
            l_mse = []
            for batch in batches:
                t_actions = torch.tensor(actions[batch])
                t_states = torch.tensor(states[batch])
                t_old_probs = torch.tensor(probs[batch])
                t_state_values = torch.tensor(state_values[batch])
                t_rewards = torch.tensor(rewards[batch])
                t_advantages = torch.tensor(advantages[batch])

                mse = self.train_step_batch(t_states,t_old_probs,t_advantages,t_state_values,t_rewards,t_actions)
                l_mse.append(mse)
        return sum(l_mse) / len(l_mse)

            
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
        state_old = agent.get_state(env = simulation)
        final_move,action_probs,state_value = agent.choose_actions(state = torch.tensor(state_old,dtype=torch.float))
        reward, done, score = simulation.step(actions = final_move)
        state_new = agent.get_state(env = simulation)
        agent.remember(state=state_old,actions=final_move,reward=reward,
                       next_state=state_new,done=done,state_value=state_value,log_probs=action_probs)
        if done:
            agent.complete_vectors()
            mse = agent.learn()
            agent.memory.clear_memory()
            simulation.reset()
            agent.n_games += 1

            if score > record:
                record = score
                # agent.model.save()
            print(f"Game:{agent.n_games}   |Score:{score}     |Record:{record}")
            plot_scores.append(score)
            plot_mse.append(mse.item())
            total_score += score
            mean_score = total_score/agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores,plot_mean_scores,plot_mse)


if __name__ == "__main__":
    train()
