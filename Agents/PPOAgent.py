import os
import numpy as np
from collections import deque
import sys

import torch
from stable_baselines3.common.env_checker import check_env

upper_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, upper_folder)
from Neural_Nets import PPO_NN,ActorNetwork,CriticalNetwork
from RL_env.Doggy_walker_env import Doggy_walker_env
from helpers.plot_helper import plot



GAMMA = 0.99
LAMBDA = 0.95
NN_ACTOR_DIMENSIONS = [512,256,256]
NN_CRITIC_DIMENSIONS = [512,256,128]
POLICY_CLIP = 0.2
LEARNING_RATE_ACTOR = 1e-4
LEARNING_RATE_CRITIC = 1e-4
EPOCHS = 10
BATCH_SIZE = 10
MSE_CTE = 0.5
ENTROPY_CTE = 0.5

SINGLE_NN = False
NORMALIZE_DATA = False
CLASSIC_RETURNS = True


class PPOMemory:
    def __init__(self,batch_size,input_size):
        self.batch_size = batch_size
        self.last_scores = deque(maxlen=100)
        self.plot_scores = []
        self.plot_mean_scores = []
        self.clear_memory()
        self.maxs = np.zeros(input_size)
        self.mins = np.zeros(input_size)
        self.n_games = 0
        self.load_registered_scores()


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

    def save_score(self,score):
        self.plot_scores.append(score)
        self.last_scores.append(score)
        self.plot_mean_scores.append(np.mean(self.last_scores))

    def plot(self):
        plot(self.plot_scores, self.last_scores, self.plot_mean_scores)
    
    def register_score(self):
        np.savez('models/registered_scores.npz', plot_scores=self.plot_scores, last_scores=list(self.last_scores),plot_mean_scores = self.plot_mean_scores)


    def load_registered_scores(self):
        if os.path.exists("models/registered_scores.npz"):
            dados = np.load('models/registered_scores.npz')
            self.plot_scores = dados['plot_scores'].tolist()
            self.last_scores = deque(dados['last_scores'].tolist(), maxlen=100)
            self.plot_mean_scores = dados['plot_mean_scores'].tolist()
        else:
            print("No score data...")


class Agent:
    def __init__(self,env,gamma,lam,policy_clip,input_size,inner_dimensions_actor,inner_dimensions_critic,n_joints,lr_actor,lr_critic,epochs,batch_size,c1,c2,single_nn,normalize_data,classic_returns):
        self.env = env
        self.gamma = gamma
        self.lam = lam
        self.policy_clip = policy_clip
        self.input_size = input_size
        self.inner_dimensions_actor = inner_dimensions_actor
        self.inner_dimensions_critic = inner_dimensions_critic
        self.n_joints = n_joints
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.epochs = epochs
        self.batch_size = batch_size
        self.memory = PPOMemory(batch_size=batch_size,input_size=input_size)
        # self.nn = PPO_NN(inner_dimensions = inner_dimensions_actor,
        #                  num_inputs = input_size,
        #                  num_outputs=n_joints)
        # self.optimizer = optim.Adam(self.nn.parameters(),lr=lr)
        self.actor = ActorNetwork(input_size = input_size,
                                  hidden_size1=inner_dimensions_actor[0],
                                  hidden_size2=inner_dimensions_actor[1],
                                  hidden_size3=inner_dimensions_actor[2],
                                  output_size = n_joints,
                                  lr = lr_actor,)
        self.critic = CriticalNetwork(input_size= input_size,
                                      hidden_size1=inner_dimensions_critic[0],
                                      hidden_size2=inner_dimensions_critic[1],
                                      hidden_size3=inner_dimensions_critic[2],
                                      output_size=1,
                                      lr=lr_critic)
        self.c1 = c1
        self.c2 = c2
        self.single_nn = single_nn
        self.normalize_data = normalize_data
        self.classic_returns = classic_returns
        self.last_observation,info = self.env.reset()
        if normalize_data:
            self.last_observation = self.state_normalization(self.last_observation)
        self.action_low = torch.tensor(env.action_space.low, dtype=torch.float32)
        self.action_high = torch.tensor(env.action_space.high, dtype=torch.float32)
        self.load_model()
        self.best_score = float("-inf")


    def train_step(self):
        if self.single_nn:
            dist, value = self.nn(torch.tensor(self.last_observation,dtype=torch.float))
            
        else:
            dist = self.actor(torch.tensor(self.last_observation,dtype=torch.float))
            value = self.critic(torch.tensor(self.last_observation,dtype=torch.float))
        
        value = torch.squeeze(value).item()
        action = dist.sample()
        log_prob = dist.log_prob(action)
        new_observation, reward, terminated,truncated, info = self.env.step(action)
        if self.normalize_data:
            new_observation = self.state_normalization(new_observation)
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
            self.memory.n_games += 1
            if not terminated:
                if self.single_nn:
                    _, next_value = self.nn(torch.tensor(new_observation, dtype=torch.float))
                else:
                    next_value = self.critic(torch.tensor(new_observation,dtype=torch.float))

            else:
                next_value = torch.tensor([0.], dtype=torch.float)
            self.learn(next_value = next_value)
            self.last_observation, info = self.env.reset()

            self.memory.save_score(info["score"])
            self.memory.plot()
            
            if info["score"] > self.best_score:
                self.best_score = info["score"]
                self.save_model()
            if self.memory.n_games % 100 == 0:
                self.save_model()




    def state_normalization(self,state):
        epsilon = 1e-8
        self.memory.mins = np.minimum(state,self.memory.maxs)
        norm_state = (state - self.memory.mins)/(self.memory.maxs-self.memory.mins+epsilon)
        return norm_state



    def learn(self,next_value = None):
        if self.classic_returns:
            self.memory.buffer_advantages,_ = self.calculate_GAE(next_value = None)
            self.memory.buffer_returns = self.calculate_returns_old_fashion(self.memory.buffer_rewards)
        else:
            self.memory.buffer_advantages,self.memory.buffer_returns = self.calculate_GAE(next_value = None)
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
                normalized_advantages = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

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
        if self.single_nn:
            dist, value = self.nn(torch.tensor(states,dtype=torch.float))
        else:
            dist = self.actor(torch.tensor(states,dtype=torch.float))
            value = self.critic(torch.tensor(states,dtype=torch.float))

        new_probs = dist.log_prob(actions)

        r_theta = torch.exp(new_probs - old_probs)

        clipped = torch.clamp(r_theta, 1-self.policy_clip, 1+self.policy_clip)*advantages.unsqueeze(1)
        not_clipped = r_theta*advantages.unsqueeze(1)

        min_loss = -torch.min(clipped, not_clipped).mean()
        entropy = dist.entropy().mean()

        policy_loss = min_loss - c2*entropy

        mse_loss = ((returns - torch.squeeze(value))**2).mean()
        loss_function = policy_loss + c1*mse_loss


        if self.single_nn:
            self.optimizer.zero_grad()
            loss_function.backward()
            self.optimizer.step()
            
        else:
            self.actor.optimizer.zero_grad()
            policy_loss.backward()
            self.actor.optimizer.step()

            self.critic.optimizer.zero_grad()
            mse_loss.backward()
            self.critic.optimizer.step()

        


    def calculate_GAE(self,next_value = None):
        rewards = self.memory.buffer_rewards
        if next_value != None:
            values = self.memory.buffer_state_values + [next_value]
        else:
            values = self.memory.buffer_state_values
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] - values[t] if t < len(rewards) - 1 else rewards[t] - values[t] #Calculo da vantagem específica do tempo T
            gae = delta + self.gamma * self.lam * gae
            advantages[t] = gae
            returns[t] = gae + values[t]

        return advantages.tolist(),returns.tolist()

    def calculate_returns_old_fashion(self,rewards):
        returns = np.zeros_like(rewards, dtype=np.float32)
        future_reward = 0.0  

        for t in reversed(range(len(rewards))):
            future_reward = rewards[t] + self.gamma * future_reward
            returns[t] = future_reward

        return returns.tolist()



    def train(self, n_steps = None):
        if n_steps == None:
            while True:
                self.train_step()
        else:
            for step in range(n_steps):
                self.train_step()

    def save_model(self):
        print("... Saving models ...")
        if self.single_nn:
            self.nn.save()
        else:
            self.actor.save()
            self.critic.save()

        self.memory.register_score()

    def load_model(self):
        if os.path.exists("models/critic_model.pth") and os.path.exists("models/actor_model.pth"):
            print("...Loading Model...")
            if self.single_nn:
                self.nn.critic.load_state_dict(torch.load("models/critic_model.pth"))
                self.nn.actor.load_state_dict(torch.load("models/actor_model.pth"))
            else:   
                self.actor.load_state_dict(torch.load("models/actor_model.pth"))
                self.critic.load_state_dict(torch.load("models/critic_model.pth"))
            
        else:
            print("File no found, no model will be loaded")


if __name__ == "__main__":

    env = Doggy_walker_env()
    check_env(env)

    model = Agent(
        gamma = GAMMA,
        lam = LAMBDA,
        policy_clip = POLICY_CLIP,
        input_size = env.observation_space.shape[0],
        inner_dimensions_actor = NN_ACTOR_DIMENSIONS,
        inner_dimensions_critic = NN_CRITIC_DIMENSIONS,
        n_joints = env.action_space.shape[0],
        lr_actor = LEARNING_RATE_ACTOR,
        lr_critic = LEARNING_RATE_CRITIC,
        epochs = EPOCHS,
        batch_size = BATCH_SIZE,
        env = env,
        c1 = MSE_CTE,
        c2 = ENTROPY_CTE,
        single_nn = SINGLE_NN,
        normalize_data = NORMALIZE_DATA,
        classic_returns = CLASSIC_RETURNS
        )

    model.train()
