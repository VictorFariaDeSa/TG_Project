import gymnasium as gym
from gymnasium import spaces
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import math
from RL_env.Doggy_robot import Doggy_robot
from helpers.coppelia_helper import create_stepped_sim,reset_sim,start_sim

MAX_STEPS = 1000
HIGH_ACTION = 3
LOW_ACTION = -3



LOW_LOWER = -math.pi*3/4
HIGH_LOWER = 0
LOW_UPPER = -math.pi/2
HIGH_UPPER = math.pi/2






class Doggy_walker_env(gym.Env):
    metadata = {"render_modes": [], "render_fps": 60}

    def __init__(self):
        super().__init__()
        low = np.array([LOW_UPPER]*4 + [LOW_LOWER]*4, dtype=np.float32)
        high = np.array([HIGH_UPPER]*4 + [HIGH_LOWER]*4, dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high,shape=(8,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32)

        self.sim = create_stepped_sim()
        
        self.last_time = self.sim.getSimulationTime()
        self.n_steps = 0
        self.score = 0

        self.robot = Doggy_robot(sim = self.sim, robot_name="Doggy",target_name="Target" )
        start_sim(self.sim)


        
    def step(self, action):
        clipped_action = np.clip(action, self.action_space.low, self.action_space.high)
        self.robot.input_position_actions(clipped_action)
        self.sim.step()
        self.robot.update_robot_data()
        self.n_steps += 1
        obs = self.get_observation()
        reward = self.get_reward(clipped_action)
        done = self.check_done()
        truncated = self.n_steps >= MAX_STEPS
        self.score += reward
        info = {}
        if truncated:
            info["end_cause"] = "truncated"
        if done:
            if self.robot.fall:
                info["end_cause"] = "fall"
            elif self.robot.upside_down:
                info["end_cause"] = "upside_down"
            elif self.robot.arrival:
                info["end_cause"] = "arrival"
        return obs, reward, bool(done), truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        info_dict = {"score": self.score}
        self.score = 0 
        self.n_steps = 0

        reset_sim(self.sim)

        self.robot.last_x = self.robot.positions[0]
        self.last_time = self.sim.getSimulationTime()
        self.robot.last_time = self.sim.getSimulationTime()

        obs = self.get_observation()
        info = info_dict

        return obs, info


    def get_reward(self, action):
        [x,y,z] = self.robot.positions
        roll, pitch, yaw = self.robot.orientations
        dx = self.robot.delta_x
        laydown = self.robot.fall
        reached = self.robot.arrival
        upside_down = self.robot.upside_down
        [vx, vy, vz] = self.robot.linear_velocities
        n_changes_joints_orientation = self.robot.get_joints_orientation_change()
        zero_speed_joints = sum(self.robot.get_joints_speed_0())
        cg_in = self.robot.cg_inside()
        poligon_area = self.robot.get_poligon_area()
        height_goal = 0.35
        height_range = abs(height_goal-z) < 0.1
        maxed_joints = self.robot.get_joints_on_max()
        n_maxed_joints = sum(maxed_joints)
        loss_angle = self.robot.get_correct_direction_angle()
        joints_accel = np.abs(self.robot.get_joints_acceleration())
        feets_above = self.robot.get_feet_above_base_link()
        elbows_above_feet = self.robot.get_num_elbows_above_feet()
        foot_distance = self.robot.get_same_side_foot_distance()
        crossing_foot = self.robot.get_crossing_foot()



        dx_bonus = +(dx) * 750.0
        speed_bonus = vx*5
        reached_bonus = reached*10000
        cg_inside_bonus = cg_in * 5
        height_range_bonus = height_range*5
        correct_direction_bonus = 10 if abs(loss_angle) < math.pi/9 else abs(loss_angle) * -1   
        area_bonus = 1 if poligon_area > 0.2 else 0
        elbows_bonus = elbows_above_feet * 0.5 if elbows_above_feet < 4 else 5
        crossing_foot_bonus = -10 * sum(x==1 for x in crossing_foot)
        foot_distance_bonus = -5 * sum(x<0.15 for x in foot_distance)


        vy_bonus = abs(vy)*-0.5
        vz_bonus = abs(vz)*-0.5
        pitch_bonus = abs(pitch) * -0.5 #era 0.5
        roll_bonus = abs(roll)*-0.5
        y_offset_bonus = abs(y) * -5
        vel_0_bonus = (abs(vx)<0.1) * -0.5
        maxed_joints_bonus = n_maxed_joints * -0.5 #era 0.5
        n_changes_joints_orientation_bonus = n_changes_joints_orientation * -0.1
        zero_speed_joints_bonus = zero_speed_joints * -0.1
        joints_accel_bonus = np.sum(joints_accel) * -0.01
        feets_above_bonus = feets_above * (-0.2)

        laydown_bonus = (laydown) * -1000.0
        upside_down_bonus = upside_down * -1000

        



        reward = (
            dx_bonus
            + speed_bonus
            + reached_bonus
            + correct_direction_bonus
            + pitch_bonus
            + roll_bonus
            + y_offset_bonus
            + vel_0_bonus
            + laydown_bonus
            + cg_inside_bonus
            + upside_down_bonus
            + area_bonus
            +height_range_bonus
            +vy_bonus
            +vz_bonus
            +maxed_joints_bonus
            +n_changes_joints_orientation_bonus
            +zero_speed_joints_bonus
            +joints_accel_bonus
            +feets_above_bonus
            +elbows_bonus
            +crossing_foot_bonus
            +foot_distance_bonus

        )

        return float(reward)

    def get_observation(self):
        obs = []

        x, y, z = self.robot.positions
        roll,pitch,yaw = self.robot.orientations
        (vx, vy, vz) = self.robot.linear_velocities
        (wx, wy, wz) = self.robot.angular_velocities
        joints_data = self.robot.get_all_joints_information()

        obs.extend([x, y, z])
        obs.extend([roll,pitch,yaw])
        obs.extend([vx, vy, vz, wx, wy, wz])
        obs.extend(joints_data)


        return np.array(obs, dtype=np.float32)

    def check_done(self):
        return self.robot.fall or self.robot.arrival or self.robot.upside_down


    

    
            


