import h5py
from helpers.plot_helper import plot
import numpy as np
from collections import deque
import os

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
        self.initialize_analysis_buffers()


    def initialize_analysis_buffers(self):
        self.initialize_positions_buffers()
        self.initialize_speeds_buffers()
        self.initialize_RPY_buffers()
        self.initialize_misc_buffers()

    def initialize_positions_buffers(self):
        self.x_pos_buffer = []
        self.y_pos_buffer = []
        self.z_pos_buffer = []
    
    def initialize_speeds_buffers(self):
        self.x_speed_buffer = []
        self.y_speed_buffer = []
        self.z_speed_buffer = []

    def initialize_RPY_buffers(self):
        self.roll_buffer = []
        self.pitch_buffer = []
        self.yaw_buffer = []

    def initialize_misc_buffers(self):
        self.poligon_area_buffer = []
        self.cg_inside_buffer = []
        self.relative_yaw_angle_buffer = []
        self.rewards_buffer = []


    def remember_all(self,positions,speeds,rpy,poligon_area,cg_inside,relative_yaw_angle,reward):
        self.remember_positions(postions=positions)
        self.remember_speeds(speeds=speeds)
        self.remember_RPY(rpy=rpy)
        self.remember_misc(poligon_area=poligon_area,
                           cg_inside=cg_inside,
                           relative_yaw_angle=relative_yaw_angle,
                           reward = reward)

    def remember_positions(self,postions):
        x,y,z = postions
        self.x_pos_buffer.append(x)
        self.y_pos_buffer.append(y)
        self.z_pos_buffer.append(z)

    def remember_speeds(self,speeds):
        vx,vy,vz = speeds
        self.x_speed_buffer.append(vx)
        self.y_speed_buffer.append(vy)
        self.z_speed_buffer.append(vz)

    def remember_RPY(self,rpy):
        roll,pitch,yaw = rpy
        self.roll_buffer.append(roll)
        self.pitch_buffer.append(pitch)
        self.yaw_buffer.append(yaw)

    def remember_misc(self,poligon_area,cg_inside,relative_yaw_angle,reward):
        self.poligon_area_buffer.append(poligon_area)
        self.cg_inside_buffer.append(cg_inside)
        self.relative_yaw_angle_buffer.append(relative_yaw_angle)
        self.rewards_buffer.append(reward)



    def save_h5_all(self):
        self.save_h5_positions()
        self.save_h5_speeds()
        self.save_h5_RPY()
        self.save_h5_misc()



    def save_losses(self,mean_loss,mean_mse):
        with h5py.File('models/Positions.h5', 'a') as f:
            loss_name = f'loss_{self.n_games}'
            mse_name = f'mse_{self.n_games}'
            if loss_name in f:
                del f[loss_name]
            if mse_name in f:
                del f[mse_name]
            f.create_dataset(loss_name, data=mean_loss)
            f.create_dataset(mse_name, data=mean_mse)

    def save_end_cause(self,info):
        with h5py.File('models/Positions.h5', 'a') as f:
            info_name = f'info_{self.n_games}'
            if info_name in f:
                del f[info_name]
            f.create_dataset(info_name, data=info)


    def save_h5_positions(self):
        with h5py.File('models/Positions.h5', 'a') as f:
            x_name = f'x_{self.n_games}'
            y_name = f'y_{self.n_games}'
            z_name = f'z_{self.n_games}'
            if x_name in f:
                del f[x_name]
            if y_name in f:
                del f[y_name]
            if z_name in f:
                del f[z_name]
            f.create_dataset(x_name, data=self.x_pos_buffer)
            f.create_dataset(y_name, data=self.y_pos_buffer)
            f.create_dataset(z_name, data=self.z_pos_buffer)
    
    def save_h5_speeds(self):
        with h5py.File('models/Speeds.h5', 'a') as f:
            x_name = f'vx_{self.n_games}'
            y_name = f'vy_{self.n_games}'
            z_name = f'vz_{self.n_games}'
            if x_name in f:
                del f[x_name]
            if y_name in f:
                del f[y_name]
            if z_name in f:
                del f[z_name]
            f.create_dataset(x_name, data=self.x_speed_buffer)
            f.create_dataset(y_name, data=self.y_speed_buffer)
            f.create_dataset(z_name, data=self.z_speed_buffer)
    
    def save_h5_RPY(self):
        with h5py.File('models/RPY.h5', 'a') as f:
            r_name = f'roll_{self.n_games}'
            p_name = f'pitch_{self.n_games}'
            y_name = f'yaw_{self.n_games}'
            if r_name in f:
                del f[r_name]
            if p_name in f:
                del f[p_name]
            if y_name in f:
                del f[y_name]
            f.create_dataset(r_name, data=self.roll_buffer)
            f.create_dataset(p_name, data=self.pitch_buffer)
            f.create_dataset(y_name, data=self.yaw_buffer)

    def save_h5_misc(self):
        with h5py.File('models/Misc.h5', 'a') as f:
            poligon_area_name = f'x_{self.n_games}'
            cg_inside_name = f'y_{self.n_games}'
            relative_angle_name = f'z_{self.n_games}'
            reward_name = f'reward_{self.n_games}'

            if poligon_area_name in f:
                del f[poligon_area_name]
            if cg_inside_name in f:
                del f[cg_inside_name]
            if relative_angle_name in f:
                del f[relative_angle_name]
            if reward_name in f:
                del f[reward_name]
            f.create_dataset(poligon_area_name, data=self.poligon_area_buffer)
            f.create_dataset(cg_inside_name, data=self.cg_inside_buffer)
            f.create_dataset(relative_angle_name, data=self.relative_yaw_angle_buffer)
            f.create_dataset(reward_name, data=self.rewards_buffer)

    

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

        self.initialize_analysis_buffers()

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
            self.n_games = len(self.plot_mean_scores)
        else:
            print("No score data...")


        