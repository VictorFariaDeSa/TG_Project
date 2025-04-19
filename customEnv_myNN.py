from stable_baselines3.common.env_checker import check_env
from customEnviroment import Doggy_walker
from torch import nn
import torch
from torch.distributions import Normal
import numpy as np
import torch.optim as optim
import os

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
    def __init__(self,input_size,inner_dimensions,output_size,lr,std):
        super().__init__()

        in_features = input_size
        actor_layers = []
        for out_features in inner_dimensions:
            actor_layers.append(nn.Linear(in_features, out_features))
            actor_layers.append(nn.Tanh())
            in_features = out_features
        
        actor_layers.append(nn.Linear(in_features, output_size))

        self.actor = nn.Sequential(*actor_layers)
        self.log_std = nn.Parameter(torch.ones(1, output_size) * std)
        self.optimizer = optim.Adam(self.parameters(),lr=lr)
        
    def forward(self, state):
        mu = self.actor(state)
        std = torch.exp(self.log_std)  # garantir que std seja positivo
        dist = Normal(mu, std)
        return dist
    
    def save(self,file_name="actor.pth"):
        model_folder_path = "./models"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path,file_name)
        torch.save(self.state_dict(),file_name)

class CriticalNetwork(nn.Module):
    def __init__(self,input_size,inner_dimensions,output_size,lr):
        super().__init__()
        in_features = input_size
        critic_layers = []
        for out_features in inner_dimensions:
            critic_layers.append(nn.Linear(in_features, out_features))
            critic_layers.append(nn.ReLU())
            in_features = out_features
        
        critic_layers.append(nn.Linear(in_features, output_size))

        self.critic = nn.Sequential(*critic_layers)
        self.optimizer = optim.Adam(self.parameters(),lr=lr)

        
    def forward(self,state):
        value = self.critic(state)
        return value

    def save(self,file_name="critic.pth"):
        model_folder_path = "./models"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path,file_name)
        torch.save(self.state_dict(),file_name)

class Agent:
    def __init__(self,env,gamma,lam,policy_clip,input_size,inner_dimensions,n_joints,lr,epochs,batch_size,c1,c2):
        self.env = env
        self.gamma = gamma
        self.lam = lam
        self.policy_clip = policy_clip
        self.input_size = input_size
        self.inner_dimensions = inner_dimensions
        self.n_joints = n_joints
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.memory = PPOMemory(batch_size=batch_size)
        self.nn = PPO_NN(inner_dimensions = inner_dimensions,
                         num_inputs = input_size,
                         num_outputs=n_joints)
        self.optimizer = optim.Adam(self.nn.parameters(),lr=lr)

        # self.actor = ActorNetwork(input_size = input_size,
        #                           inner_dimensions = inner_dimensions,
        #                           output_size = n_joints,
        #                           lr = lr,
        #                           std=0.0)
        # self.critic = CriticalNetwork(input_size= input_size,
        #                               inner_dimensions= inner_dimensions,
        #                               output_size=1,
        #                               lr=lr)
        self.c1 = c1
        self.c2 = c2
        self.last_observation,info = self.env.reset()
        self.action_low = torch.tensor(env.action_space.low, dtype=torch.float32)
        self.action_high = torch.tensor(env.action_space.high, dtype=torch.float32)
        self.load_model()
        self.best_score = float("-inf")


    def train_step(self):
        dist, value = self.nn(torch.tensor(self.last_observation,dtype=torch.float))
        # dist = self.actor(torch.tensor(self.last_observation,dtype=torch.float))
        # value = self.critic(torch.tensor(self.last_observation,dtype=torch.float))
        value = value.item()
        action = dist.sample().squeeze(0)
        log_prob = dist.log_prob(action).squeeze()
        action = action.detach().numpy()
        new_observation, reward, terminated,truncated, info = self.env.step(action)
        self.memory.remember(
            state = self.last_observation,
            actions = action,
            reward = reward,
            done = terminated,
            next_state = new_observation,
            state_value = value,
            log_probs = log_prob.detach().numpy()
        )
        self.last_observation = new_observation
        if terminated or truncated:
            if not terminated:
                _, next_value = self.nn(torch.tensor(new_observation, dtype=torch.float))
            else:
                next_value = torch.tensor([0.], dtype=torch.float)
            # next_value = self.critic(torch.tensor(new_observation,dtype=torch.float))
            self.learn(next_value = next_value)
            self.last_observation, info = self.env.reset()
            if info["score"] > self.best_score:
                self.best_score = info["score"]
                self.save_model()


    def learn(self,next_value = None):
        self.memory.buffer_advantages,self.memory.buffer_returns = self.calculate_GAE(next_value = next_value)
        for epoch in range(self.epochs):
            states,actions,rewards,terminals,\
            next_states,probs,state_values,\
            advantages, returns, batches = self.memory.generate_batches()
            for batch in batches:
                if batch.size == 1:
                    continue
                states_batch = torch.tensor(states[batch])
                actions_batch = torch.tensor(actions[batch])
                rewards_batch = torch.tensor(rewards[batch])
                next_states_batch = torch.tensor(next_states[batch])
                old_probs_batch = torch.tensor(probs[batch])
                state_values_batch = torch.tensor(state_values[batch])
                advantages_batch = torch.tensor(advantages[batch])
                returns_batch = torch.tensor(returns[batch])
                terminal_batch = torch.tensor(terminals[batch])

                self.train_step_batch(states = states_batch,
                                      actions = actions_batch,
                                      rewards = rewards_batch,
                                      next_states = next_states_batch,
                                      terminals = terminal_batch,
                                      returns = returns_batch,
                                      old_probs = old_probs_batch,
                                      state_values = state_values_batch,
                                      advantages = advantages_batch,
                                      c1 = self.c1,
                                      c2 = self.c2
                                      )

        self.memory.clear_memory()

    def train_step_batch(self,states,actions,rewards,next_states,terminals,returns,old_probs,state_values,advantages,c1,c2):
        dist, value = self.nn(torch.tensor(states,dtype=torch.float))
        # dist = self.actor(torch.tensor(states,dtype=torch.float))
        # value = self.critic(torch.tensor(states,dtype=torch.float))
        new_probs = dist.log_prob(actions)

        r_theta = torch.exp(new_probs - old_probs).squeeze()
        clipped = torch.clamp(r_theta, 1-self.policy_clip, 1+self.policy_clip)*advantages.unsqueeze(1)
        not_clipped = r_theta*advantages.unsqueeze(1)

        min_loss = -torch.min(clipped, not_clipped).mean()
        entropy = dist.entropy().mean()

        policy_loss = min_loss - c2*entropy

        mse_loss = ((returns - value)**2).mean()
        loss_function = min_loss + c1*mse_loss - c2*entropy



        self.optimizer.zero_grad()
        loss_function.backward()
        self.optimizer.step()

        # self.actor.optimizer.zero_grad()
        # policy_loss.backward()
        # self.actor.optimizer.step()

        # self.critic.optimizer.zero_grad()
        # mse_loss.backward()
        # self.critic.optimizer.step()


    def calculate_GAE(self,next_value = None):
        rewards = self.memory.buffer_rewards
        if next_value != None:
            values = self.memory.buffer_state_values + [next_value]
        else:
            values = self.memory.buffer_state_values
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(self.memory.buffer_state_values))):
            mask = (1-self.memory.buffer_is_terminal[t])
            delta = rewards[t] + self.gamma * values[t + 1] * mask - values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages.tolist(),returns.tolist()

    def calculate_returns(self):
        returns = (np.array(self.memory.buffer_advantages) + np.array(self.memory.buffer_state_values)).tolist()
        return returns




    def train(self, n_steps = None):
        if n_steps == None:
            while True:
                self.train_step()
        else:
            for step in range(n_steps):
                self.train_step()

    def save_model(self):
        print("... Saving models ...")
        self.nn.save()
        # self.actor.save()
        # self.critic.save()

    def load_model(self):
        if os.path.exists("models/critic_model.pth") and os.path.exists("models/actor_model.pth"):
            print("...Loading Model...")
            self.nn.critic.load_state_dict(torch.load("models/critic_model.pth"))
            self.nn.actor.load_state_dict(torch.load("models/actor_model.pth"))
            # self.actor.load_state_dict(torch.load("models/actor.pth"))
            # self.critic.load_state_dict(torch.load("models/critic.pth"))

        else:
            print("File no found, no model will be loaded")


if __name__ == "__main__":
    env = Doggy_walker()
    check_env(env)

    model = Agent(
        gamma = 0.99,
        lam = 0.95,
        policy_clip = 0.2,
        input_size = env.observation_space.shape[0],
        inner_dimensions = [216, 216, 128],
        n_joints = env.action_space.shape[0],
        lr = 1e-4,
        epochs = 5,
        batch_size = 64,
        env = env,
        c1 = 0.5,
        c2 = 0.001
        )

    model.train()
