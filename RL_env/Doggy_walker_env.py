import gymnasium as gym
from gymnasium import spaces
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import math
from RL_env.Doggy_robot import Doggy_robot
from helpers.coppelia_helper import create_stepped_sim,reset_sim,start_sim

MAX_STEPS = 1000
HIGH_ACTION = 1
LOW_ACTION = -1

class Doggy_walker_env(gym.Env):
    metadata = {"render_modes": [], "render_fps": 60}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=LOW_ACTION, high=HIGH_ACTION, shape=(8,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(31,), dtype=np.float32)

        self.sim = create_stepped_sim()
        
        self.last_time = self.sim.getSimulationTime()
        self.n_steps = 0
        self.score = 0

        self.robot = Doggy_robot(sim = self.sim, robot_name="Doggy",target_name="Target" )
        start_sim(self.sim)


        
    def step(self, action):
        actions = np.clip(action,LOW_ACTION,HIGH_ACTION)
        self.robot.input_speed_actions(actions)
        self.sim.step()
        self.n_steps += 1
        obs = self.get_observation()
        reward = self.get_reward(actions)
        done = self.check_done()
        truncated = self.n_steps >= MAX_STEPS

        self.score += reward
        info = {}

        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        info_dict = {"score": self.score}
        self.score = 0 
        self.n_steps = 0

        reset_sim(self.sim)

        self.robot.last_x = self.robot.get_relative_position()[0]
        self.last_time = self.sim.getSimulationTime()

        obs = self.get_observation()
        info = info_dict

        return obs, info


    def get_reward(self, action):
        [x,y,z] = self.robot.get_relative_position()
        roll, pitch, yaw = self.robot.get_orientation()
        stable = self.robot.check_stability(math.radians(45))
        effort = sum(abs(action))
        dx = self.robot.get_delta_x()
        laydown = self.robot.check_fall()
        reached = self.robot.check_arrival()
        [vx, vy, vz], [wx, wy, wz] = self.robot.get_velocities()
        lazy_joints = (np.abs(self.robot.get_joints_angle_change()) < 0.05).all()


        height_goal = 0.35

        dx_bonus = +(dx) * 1000.0
        height_bonus = +(z>0.25) * 1.0
        laydown_bonus = -(laydown) * 1000.0
        stable_bonus = (1 if stable else -1) * 1.0
        pitch_bonus = - (abs(pitch)) * 1.0
        progress_bonus = + (25-abs(x)) * 5
        speed_bonus = + (vx) * 5
        yaw_bonus = - (abs(yaw)) * 1 
        stagnated_bonus = -(abs(vx)<0.1) * 3
        lazy_joints_bonus = -1 if lazy_joints else 0





        reward = (
            dx_bonus
            # + height_bonus
            + laydown_bonus
            # + stable_bonus
            # + pitch_bonus
            + progress_bonus
            + speed_bonus
            # + yaw_bonus
            # + stagnated_bonus
        )

        return float(reward)


    def get_observation(self):
        obs = []

        x, y, z = self.robot.get_relative_position()
        roll,pitch,yaw = self.robot.get_orientation()
        (vx, vy, vz), (wx, wy, wz) = self.robot.get_velocities()
        joints_data = self.robot.get_all_joints_information()

        obs.extend([x, y, z])
        obs.extend([roll,pitch,yaw])
        obs.extend([vx, vy, vz, wx, wy, wz])
        obs.extend([vx,vy,vz])
        obs.extend(joints_data)


        return np.array(obs, dtype=np.float32)

    def check_done(self):
        return self.robot.check_fall() or self.robot.check_arrival() or not self.robot.cg_inside(tolerance=1)


    

    
            


