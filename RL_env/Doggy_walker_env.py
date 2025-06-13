import gymnasium as gym
from gymnasium import spaces
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import math
from RL_env.Doggy_robot import Doggy_robot
from helpers.coppelia_helper import create_stepped_sim,reset_sim,start_sim
import random
import pickle
import os


CONTROL = "Position"


MAX_STEPS = 1000

if CONTROL == "Position":
    LOW_LOWER = -math.pi*3/4
    HIGH_LOWER = 0
    LOW_UPPER = -math.pi/2
    HIGH_UPPER = math.pi/2
elif CONTROL == "Speed":
    LOW_LOWER = -math.pi/2
    HIGH_LOWER = math.pi/2
    LOW_UPPER = -math.pi/2
    HIGH_UPPER = math.pi/2
elif CONTROL == "Force":
    LOW_LOWER = -10000
    HIGH_LOWER = 10000
    LOW_UPPER = -10000
    HIGH_UPPER = 10000














class Doggy_walker_env(gym.Env):
    metadata = {"render_modes": [], "render_fps": 60}

    def __init__(self):
        super().__init__()
        low = np.array([LOW_UPPER]*4 + [LOW_LOWER]*4, dtype=np.float32)
        high = np.array([HIGH_UPPER]*4 + [HIGH_LOWER]*4, dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high,shape=(8,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(32,), dtype=np.float32)

        self.sim = create_stepped_sim()
        
        self.last_time = self.sim.getSimulationTime()
        self.n_steps = 0
        self.score = 0

        self.robot = Doggy_robot(sim = self.sim, robot_name="Doggy",target_name="Target" )

        self.observations = []
        self.actions = []
        self.target_pos = -9
        self.n_games = 0
        start_sim(self.sim)


        
    def step(self, action):
        clipped_action = np.clip(action, self.action_space.low, self.action_space.high)
        # controlled_actions = self.robot.convert_thetas_dict_to_actions_list(self.robot.get_thetas_dict())
        self.robot.input_position_actions(clipped_action)
        self.sim.step()
        self.robot.update_robot_data()
        self.n_steps += 1
        obs = self.get_observation()
        reward = self.get_reward(clipped_action)
        done = self.check_done()
        # self.observations.append(obs)
        # self.actions.append(controlled_actions)
        # print(len(self.observations))
        # if len(self.observations) == 50000:
        #     dataset = {
        #         "observations": np.array(self.observations),
        #         "actions": np.array(self.actions)
        #     }

        #     os.makedirs("dataset", exist_ok=True)
        #     with open("dataset/imitation_data.pkl", "wb") as f:
        #         pickle.dump(dataset, f)

        #     print("✅ Dataset salvo com sucesso.")
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
        self.n_games += 1
        if self.n_games % 250 == 0 and self.target_pos < 10:
            self.target_pos += 0.5
        reset_sim(self.sim)
        y = random.uniform(-2, 2)
        self.sim.setObjectPosition(self.robot.robot, -1, [-12, y, 0.4])
        self.sim.setObjectPosition(self.robot.target, -1, [self.target_pos, 0, 0.002])
        self.robot.last_x = self.robot.positions[0]
        self.last_time = self.sim.getSimulationTime()
        self.robot.last_time = self.sim.getSimulationTime()

        obs = self.get_observation()
        info = info_dict
        

        return obs, info


    # def get_reward(self, action):
    #     [x,y,z] = self.robot.positions
    #     height = self.robot.height
    #     roll, pitch, yaw = self.robot.orientations
    #     dx = self.robot.delta_x
    #     laydown = self.robot.fall
    #     reached = self.robot.arrival
    #     upside_down = self.robot.upside_down
    #     [vx, vy, vz] = self.robot.linear_velocities
    #     [wx,wy,wz] = self.robot.angular_velocities
    #     n_changes_joints_orientation = self.robot.get_joints_orientation_change()
    #     zero_speed_joints = sum(self.robot.get_joints_speed_0())
    #     cg_in = self.robot.cg_inside()
    #     poligon_area = self.robot.get_poligon_area()
    #     height_goal = 0.35
    #     height_range = abs(height_goal-z) < 0.1
    #     maxed_joints = self.robot.get_joints_on_max()
    #     n_maxed_joints = sum(maxed_joints)
    #     loss_angle = self.robot.get_correct_direction_angle()
    #     joints_accel = np.abs(self.robot.get_joints_acceleration())
    #     feets_above = self.robot.get_feet_above_base_link()
    #     elbows_above_feet = self.robot.get_num_elbows_above_feet()
    #     foot_distance = self.robot.get_same_side_foot_distance()
    #     crossing_foot = self.robot.get_crossing_foot()
    #     stationary_joints = self.robot.get_delta_foot_positions()
    #     delta_positions = self.robot.get_joints_delta_x_pos()
    #     sum_delta_positions = sum(abs(x) for x in delta_positions)




    #     correct_direction_bonus = 10 if abs(loss_angle) < math.pi/9 else abs(loss_angle) * -1   
    #     elbows_bonus = elbows_above_feet * 0.5 if elbows_above_feet < 4 else 5
    #     crossing_foot_bonus = -10 * sum(x==1 for x in crossing_foot)
    #     foot_distance_bonus = -5 * sum(x<0.15 for x in foot_distance)




    #     y_offset_bonus = abs(y) * -5
    #     vel_0_bonus = (abs(vx)<0.1) * -0.5
    #     maxed_joints_bonus = n_maxed_joints * -0.5 #era 0.5
    #     n_changes_joints_orientation_bonus = n_changes_joints_orientation * -1
    #     stationary_joints_bonus =  stationary_joints * -5 if stationary_joints < 0.2 else 0
    #     zero_speed_joints_bonus = zero_speed_joints * -0.1
    #     joints_accel_bonus = np.sum(joints_accel) * -0.01
    #     feets_above_bonus = feets_above * (-0.2)
    #     delta_pos_bonus = -2 * sum_delta_positions


        
    #     foot_colision = self.robot.foot_colision

    #     if (
    #         (foot_colision["RL"] == foot_colision["FR"]) and
    #         (foot_colision["RR"] == foot_colision["FL"]) and
    #         (foot_colision["RL"] != foot_colision["FL"]) and
    #         (
    #             (foot_colision["RL"] == 1 and self.robot.time % 1 > 0.5) or
    #             (foot_colision["RR"] == 1 and self.robot.time % 1 < 0.5)
    #         )
    #     ):
    #         walk_pattern = 1
    #     else:
    #         walk_pattern = 0


    #     reward = (
    #         + (dx * 1000)
    #         + (vx * 10)
    #         + (reached * 10000)
    #         + (10 if abs(loss_angle) < math.pi/9 else abs(loss_angle) * -1)
    #         # + (walk_pattern * 5)
            
    #         + (cg_in * 5)
    #         + ((poligon_area > 0.2) * 1)

    #         - (laydown * 2000)
    #         - (upside_down * 2000)
    #         - (abs(pitch) * 5)
    #         - (abs(roll) * 5)
    #         - (abs(vy) * 0.5)
    #         - (abs(vz) * 0.5)
    #         - (abs(y) * 5)
    #         - (abs(height-height_goal) * 5)

    #         +maxed_joints_bonus
    #         +feets_above_bonus
    #         +elbows_bonus
    #         +crossing_foot_bonus
    #         +foot_distance_bonus

    #     )


    #     # reward = (
    #     #     - abs(height_goal - z)
    #     #     - (abs(pitch) * 2)
    #     #     - (abs(yaw) * 2)
    #     #     - (abs(loss_angle) * 2)
    #     #     # - (abs(vx) * 5)
    #     #     # - (abs(vy) * 5)
    #     #     # - (abs(vz) * 5)
    #     #     # - (abs(wx) * 10)
    #     #     # - (abs(wy) * 10)
    #     #     # - (abs(wz) * 10)
    #     #     - (((abs(vx)+abs(vy)+abs(vz)) < 0.1) * 50)
    #     #     - (laydown * 10000)
    #     #     - (upside_down * 10000)
    #     #     - (n_maxed_joints * 10)
    #     #     - (feets_above * 5)
    #     #     + (cg_in * 10)
    #     #     + (poligon_area > 0.2 * 10)
    #     #     + (elbows_above_feet *5)
    #     # )
    #     # thetas = self.robot.get_thetas_dict()
    #     # programmed_action = self.robot.convert_thetas_dict_to_actions_list(thetas)

    #     # r1 = - 10 * abs((1.5 - vx))
    #     # r2 = - 0.001 * np.sum(np.array(self.robot.get_joints_torque())**2)
    #     # r3 = - 0.03 * (
    #     #         np.sum((self.robot.joints_target - self.robot.last_joints_target)**2) +
    #     #         np.sum((self.robot.joints_target - 2*self.robot.last_joints_target + self.robot.last_last_joints_target)**2)
    #     #     )
    #     # r4 = - 40 * (abs(y))
    #     # r5 = - 25 * abs(loss_angle)

    #     # if upside_down or laydown:
    #     #     print("--------------hopa ---------------")

    #     # print(f"r1: {r1} - r2: {r2} - r3: {r3} - r4: {r4} - r5: {r5}")

    #     # reward = (
    #     #     +r1 + r2 + r3 + r4 + r5
    #     #     - 100000 * upside_down
    #     #     - 100000 * laydown
    #     #     + 1000 * reached
    #     # )




    #     return float(reward)


    def get_reward(self, action):

        DESIRED_SPEED = 1
        DESIRED_HEIGHT = 0.30

        [x,y,z] = self.robot.positions
        [vx, vy, vz] = self.robot.linear_velocities
        maxed_joints = self.robot.get_joints_on_max()
        n_maxed_joints = sum(maxed_joints)
        roll, pitch, yaw = self.robot.orientations
        dx = self.robot.delta_x
        laydown = self.robot.fall
        upside_down = self.robot.upside_down
        elbows_above_feet = self.robot.get_num_elbows_above_feet()
        feets_above = self.robot.get_feet_above_base_link()
        cg_in = self.robot.cg_inside()
        loss_angle = self.robot.get_correct_direction_angle()
        foot_distance = self.robot.get_same_side_foot_distance()
        crossing_foot = self.robot.get_crossing_foot()


        r1 = abs(DESIRED_SPEED - vx)
        r2 = abs(DESIRED_HEIGHT - z)
        r3 = abs(0 - y)
        r4 = abs(feets_above)
        r5 = abs(4 - elbows_above_feet)
        r6 = (1 - cg_in)
        r7 = 0.001 * np.sum(np.array(self.robot.get_joints_torque())**2)
        r8 = 0.03 * (
                np.sum((self.robot.joints_target - self.robot.last_joints_target)**2) +
                np.sum((self.robot.joints_target - 2*self.robot.last_joints_target + self.robot.last_last_joints_target)**2)
            )
        r9 = (n_maxed_joints)
        r10 = abs(pitch) + abs(roll)
        r11 = abs(5-dx)
        r12 = (laydown or upside_down)
        r13 = abs(loss_angle)
        r14 = sum(x<0.15 for x in foot_distance)
        r15 = sum(crossing_foot)


        rewards = [
            -10 * r1,
            -25 * r2,
            -5 * r3,
            -1 * r4,
            -5 * r5,
            -1 * r6,
            -0.5 * r7,
            -5 * r8,
            -5 * r9,
            -10 * r10,
            -1 * r11,
            -50000 * r12,
            -1 * r13,
            -5 * r14,
            -10 * r15 
        ]
        print(rewards)

        return float(sum(rewards))

    def get_observation(self):
        obs = []

        x, y, z = self.robot.positions
        roll,pitch,yaw = self.robot.orientations
        (vx, vy, vz) = self.robot.linear_velocities
        (wx, wy, wz) = self.robot.angular_velocities
        joints_data = self.robot.get_all_joints_information()

        obs.extend([y, z])
        obs.extend([roll,pitch,yaw])
        obs.extend([vx, vy, vz, wx, wy, wz])
        obs.extend(joints_data)
        obs.extend([self.robot.foot_colision["RL"],self.robot.foot_colision["RR"],self.robot.foot_colision["FR"],self.robot.foot_colision["FL"]])

        for i,value in enumerate(obs):
            reading_error = random.gauss(0, 0.05)
            obs[i] = value * (1+ reading_error)

        obs.extend([self.robot.time])
        return np.array(obs, dtype=np.float32)

    def check_done(self):
        return self.robot.fall or self.robot.arrival or self.robot.upside_down


    

    
            


