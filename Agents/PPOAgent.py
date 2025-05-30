import os
import numpy as np
import sys
import torch
from stable_baselines3.common.env_checker import check_env

upper_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, upper_folder)
from Neural_Nets import PPO_NN,ActorNetwork,CriticalNetwork
from RL_env.Doggy_walker_env import Doggy_walker_env
from PPOMemory import PPOMemory


#NOTES: 
# velocidade máxima das juntas /2
# Adição de foot crossing
# adição de foot distance

GAMMA = 0.99
LAMBDA = 0.95
NN_ACTOR_DIMENSIONS = [512,512,256]
NN_CRITIC_DIMENSIONS = [512,256,128]
POLICY_CLIP = 0.2
LEARNING_RATE_ACTOR = 1e-5
LEARNING_RATE_CRITIC = 1e-5
EPOCHS = 10
BATCH_SIZE = 50
MSE_CTE = 0.5
ENTROPY_CTE = 0.01

SINGLE_NN = False
NORMALIZE_DATA = False
CLASSIC_RETURNS = False
ADVANTAGE_NORMALIZATION = True


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
        model_folder_path = "./models"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)


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
        self.memory.remember_all(
            positions=self.env.robot.positions,
            speeds=self.env.robot.linear_velocities,
            rpy = self.env.robot.orientations,
            poligon_area=self.env.robot.get_poligon_area(),
            cg_inside=self.env.robot.cg_inside(),
            relative_yaw_angle=self.env.robot.get_correct_direction_angle(),
            reward = reward)

        if terminated or truncated:
            self.memory.n_games += 1
            if not terminated:
                if self.single_nn:
                    _, next_value = self.nn(torch.tensor(new_observation, dtype=torch.float))
                else:
                    next_value = self.critic(torch.tensor(new_observation,dtype=torch.float))

            else:
                next_value = torch.tensor([0.], dtype=torch.float)
            
            self.memory.save_h5_all()
            self.memory.save_end_cause(info["end_cause"])
            mean_mse,mean_loss = self.learn(next_value = next_value)
            self.last_observation, info = self.env.reset()
            self.memory.save_losses(mean_loss,mean_mse)

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
        loss_function_list = []
        mse_loss_list = []
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
                if ADVANTAGE_NORMALIZATION:
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

                mse_loss,loss_function = self.train_step_batch(states = states_batch,
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
                mse_loss_list.append(mse_loss.item())
                loss_function_list.append(loss_function.item())

        self.memory.clear_memory()
        return np.mean(mse_loss_list),np.mean(loss_function_list)

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
        
        return mse_loss, policy_loss

        


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
                # if self.memory.n_games > 3001:
                #     break
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
