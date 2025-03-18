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
        # Initial learning rates
        self.actor_lr = 1e-4
        self.critic_lr = 3e-4
        
        self.critical_network = CriticalNetwork(
            input_size = 21,
            hidden_size1 = 512,
            hidden_size2 = 256,
            hidden_size3 = 256,
            output_size = 1,
            lr=self.critic_lr)
        self.actor_networks = [ActorNetwork(input_size=input_size,
            hidden_size1 = 512,
            hidden_size2 = 256,
            hidden_size3 = 256,
            output_size = output_size,
            lr=self.actor_lr) for _ in range(n_joints)]
        self.gamma = gamma
        self.clip_epsilon = policy_clip
        self.batch_size = batch_size
        self.memory = PPOMemory(batch_size=self.batch_size)
        self.N = N
        self.n_epochs = epochs
        self.n_games = 0
        
        # Add entropy coefficient for exploration
        self.entropy_coef = 0.01
        
        # Learning rate decay
        self.lr_decay = 0.9995

    def save_models(self):
        print("... Saving models ...")
        model_folder_path = "./models"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        # Save all actor networks
        for i, actor_network in enumerate(self.actor_networks):
            file_name = os.path.join(model_folder_path, f"actor_model{i}.pth")
            torch.save(actor_network.state_dict(), file_name)
        
        # Save critic network
        critic_file = os.path.join(model_folder_path, "critic_model.pth")
        torch.save(self.critical_network.state_dict(), critic_file)

    def load_models(self):
        print("... Loading models ...")
        model_folder_path = "./models"
        
        # Load all actor networks
        for i, actor_network in enumerate(self.actor_networks):
            file_name = os.path.join(model_folder_path, f"actor_model{i}.pth")
            if os.path.exists(file_name):
                actor_network.load_state_dict(torch.load(file_name))
                actor_network.eval()
        
        # Load critic network
        critic_file = os.path.join(model_folder_path, "critic_model.pth")
        if os.path.exists(critic_file):
            self.critical_network.load_state_dict(torch.load(critic_file))
            self.critical_network.eval()

    def get_state(self,env):
        return env.getObservation()

    def remember(self,state,actions,reward,next_state,done,log_probs,state_value):
        self.memory.remember(state,actions,reward,next_state,done,log_probs,state_value)

    def calculate_returns(self,rewards):
        for t in range(len(rewards)):
            discount = 1
            discounted_reward = 0
            for next_t in range(t,len(rewards)):
                discounted_reward += rewards[next_t]*discount
                discount *= self.gamma
            self.memory.buffer_returns.append(discounted_reward)
    


    def calculate_advantages_GAE(self, lam=0.95):
        rewards = self.memory.buffer_rewards
        values = self.memory.buffer_state_values
        
        # Create a padded array of values with an extra zero at the end
        padded_values = values + [0]
        
        # Initialize advantages array
        advantages = np.zeros_like(rewards, dtype=float)
        
        # Calculate GAE
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * padded_values[t + 1] - padded_values[t]
            gae = delta + self.gamma * lam * gae
            advantages[t] = gae  # Store just the GAE (removed adding values[t])
        
        # Normalize advantages for better training stability
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        self.memory.advantages = advantages.tolist()
        
        # Recompute returns based on advantages and values
        returns = advantages + np.array(values)
        self.memory.buffer_returns = returns.tolist()

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
        self.calculate_returns(rewards=self.memory.buffer_rewards)
        self.calculate_advantages_GAE(lam=0.95)

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
            
            # Add entropy bonus to encourage exploration
            dist = network(states.to(torch.float))
            entropy = dist.entropy().mean()
            actor_loss = l_clip - self.entropy_coef * entropy

            network.optimizer.zero_grad() 
            actor_loss.backward()
            network.optimizer.step()

        critic_value = torch.squeeze(self.critical_network(states.to(torch.float32)))
        
        critic_loss = F.mse_loss(critic_value, returns.to(torch.float32))
        self.critical_network.optimizer.zero_grad()
        critic_loss.backward()
        self.critical_network.optimizer.step()
        return critic_loss

    def learn(self):
        # Apply learning rate decay
        if self.n_games > 0 and self.n_games % 5 == 0:
            self.actor_lr *= self.lr_decay
            self.critic_lr *= self.lr_decay
            
            # Update optimizers with new learning rates
            for network in self.actor_networks:
                for param_group in network.optimizer.param_groups:
                    param_group['lr'] = self.actor_lr
                    
            for param_group in self.critical_network.optimizer.param_groups:
                param_group['lr'] = self.critic_lr
        
        total_mse = 0
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
            total_mse += sum(l_mse) / len(l_mse)
        
        avg_mse = total_mse / self.n_epochs
        return avg_mse

            
def train():
    plot_scores = []
    plot_mean_scores = []
    plot_mse = []
    total_score = 0
    record = float("-inf")
    agent = Agent()
    simulation = createSimulation("cpu")
    simulation.reset()
    
    # For tracking improvement
    no_improvement_count = 0
    best_avg_score = float("-inf")
    
    episode_count = 0
    max_episodes = 1000  # Set a maximum number of episodes
    
    while episode_count < max_episodes:
        state_old = agent.get_state(env = simulation)
        final_move, action_probs, state_value = agent.choose_actions(state = torch.tensor(state_old, dtype=torch.float))
        reward, done, score = simulation.step(actions = final_move)
        state_new = agent.get_state(env = simulation)
        
        agent.remember(state=state_old, actions=final_move, reward=reward,
                      next_state=state_new, done=done, state_value=state_value, log_probs=action_probs)
        
        if done:
            agent.complete_vectors()
            mse = agent.learn()
            agent.memory.clear_memory()
            simulation.reset()
            agent.n_games += 1
            episode_count += 1

            # Save best model
            if score > record:
                record = score
                agent.save_models()
                print(f"New record: {record}")
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Calculate running average over last 100 episodes
            plot_scores.append(score)
            plot_mse.append(mse.item())
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            
            # Check if recent average performance is improving
            if len(plot_scores) >= 100:
                recent_avg = sum(plot_scores[-100:]) / 100
                if recent_avg > best_avg_score:
                    best_avg_score = recent_avg
                    print(f"New best average: {best_avg_score}")
                    no_improvement_count = 0
            
            print(f"Game:{agent.n_games}   |Score:{score:.2f}   |Record:{record:.2f}   |Mean:{mean_score:.2f}   |MSE:{mse.item():.6f}")
            
            # Early stopping if no improvement for many episodes
            if no_improvement_count >= 100:
                print("No improvement for 100 episodes. Early stopping.")
                break
            
            # Plot progress
            if agent.n_games % 10 == 0:
                plot(plot_scores, plot_mean_scores, plot_mse)


if __name__ == "__main__":
    train()
