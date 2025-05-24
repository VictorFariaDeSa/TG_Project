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
            if self.robot.check_fall():
                info["end_cause"] = "fall"
            elif self.robot.check_upside_down():
                info["end_cause"] = "upside_down"
            elif self.robot.check_arrival():
                info["end_cause"] = "arrival"
        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        info_dict = {"score": self.score}
        self.score = 0 
        self.n_steps = 0

        reset_sim(self.sim)

        self.robot.last_x = self.robot.get_relative_position()[0]
        self.last_time = self.sim.getSimulationTime()
        self.robot.last_time = self.sim.getSimulationTime()

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
        n_changes_joints_orientation = self.robot.get_joints_orientation_change()
        zero_speed_joints = sum(self.robot.get_joints_speed_0())
        cg_in = self.robot.cg_inside()
        upside_down = self.robot.check_upside_down()
        poligon_area = self.robot.get_poligon_area()
        height_goal = 0.35
        height_range = abs(height_goal-z) < 0.1
        maxed_joints = self.robot.get_joints_on_max()
        n_maxed_joints = sum(maxed_joints)
        loss_angle = self.robot.get_correct_direction_angle()
        joints_accel = np.abs(self.robot.get_joints_acceleration())
        feets_above = self.robot.get_feet_above_base_link()

        dx_bonus = +(dx) * 750.0
        speed_bonus = vx*5
        reached_bonus = reached*10000
        cg_inside_bonus = cg_in * 5
        height_range_bonus = height_range*5
        correct_direction_bonus = 10 if abs(loss_angle) < math.pi/9 else abs(loss_angle) * -1   
        area_bonus = 1 if poligon_area > 0.2 else 0



        vy_bonus = abs(vy)*-0.5
        vz_bonus = abs(vz)*-0.5
        pitch_bonus = abs(pitch) * -5 #era 0.5
        roll_bonus = abs(roll)*-0.5
        y_offset_bonus = abs(y) * -5
        vel_0_bonus = (abs(vx)<0.1) * -0.5
        maxed_joints_bonus = n_maxed_joints * -1 #era 0.5
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
        obs.extend(joints_data)
        matrix = self.robot.get_matrix()
        # obs.append(matrix[0])


        return np.array(obs, dtype=np.float32)

    def check_done(self):
        return self.robot.check_fall() or self.robot.check_arrival() or self.robot.check_upside_down() #or not self.robot.cg_inside(tolerance=1)


    

    
            


