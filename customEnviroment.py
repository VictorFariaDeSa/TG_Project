import gymnasium as gym
from gymnasium import spaces
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
from helper import plot
from collections import deque
import math

MAX_STEPS = 1000

class Doggy_walker(gym.Env):
    metadata = {"render_modes": [], "render_fps": 60}

    def __init__(self):
        super().__init__()
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(29,), dtype=np.float32)

        # Conexão com CoppeliaSim
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')

        self.robot = self.sim.getObject('/Doggy/')
        self.target = self.sim.getObject('/Target')

        self.joint_list = [
            "RR_upper_leg_joint", "RL_upper_leg_joint",
            "FR_upper_leg_joint", "FL_upper_leg_joint",
            "RR_lower_leg_joint", "RL_lower_leg_joint",
            "FR_lower_leg_joint", "FL_lower_leg_joint"
        ]

        self.handleDict = self.getHandles()
        self.fill_robot_data()

        self.sim.setStepping(True)
        self.sim.startSimulation()

        # Inicialização
        self.last_x = self.sim.getObjectPosition(self.robot, self.target)[0]
        self.last_time = self.sim.getSimulationTime()
        self.n_steps = 0
        self.score = 0
        self.last_scores = deque(maxlen=100)
        self.plot_scores = []
        self.plot_mean_scores = []

        



    def step(self, action):
        actions  = np.clip(action, self.action_space.low, self.action_space.high)
        for i, jointName in enumerate(self.joint_list):
            joint = self.handleDict[jointName]
            self.sim.setJointTargetVelocity(joint, float(actions[i]))

        self.sim.step()
        self.n_steps += 1
        obs = self.get_observation()
        reward = self.get_reward(action)
        done = self.check_done()
        truncated = self.n_steps >= MAX_STEPS

        self.score += reward
        info = {}

        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.plot_scores.append(self.score)
        self.last_scores.append(self.score)
        self.plot_mean_scores.append(np.mean(self.last_scores))
        info_dict = {"score": self.score}
        self.score = 0 
        self.n_steps = 0

        self.sim.stopSimulation()
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            time.sleep(0.1)

        self.sim.setStepping(True)
        self.sim.startSimulation()

        self.last_x = self.sim.getObjectPosition(self.robot, self.target)[0]
        self.last_time = self.sim.getSimulationTime()

        plot(self.plot_scores, self.last_scores)

        obs = self.get_observation()
        info = info_dict

        return obs, info

    def getHandles(self):
        handleDict = {}
        for name in self.joint_list:
            handleDict[name] = self.sim.getObject(f"/Doggy/{name}")
        return handleDict

    def get_reward(self, action):
        obs = self.get_observation()
        dx =  abs(self.last_x) - abs(obs[0])
        self.last_x = obs[0]
        height = obs[2]

        roll, pitch, yaw = self.sim.getObjectOrientation(self.robot, -1)
        max_angle = math.pi / 6  # 45 graus de tolerância
        stable = abs(roll) < max_angle and abs(pitch) < max_angle and abs(yaw) < max_angle

        fall_penalty = 1000 if self.checkFall() else 0

        instability_penalty = 1 if not stable else 0

        ideal_height = 0.35
        height_penalty = abs(ideal_height - height)*0.1

        centering_bonus = max(0, 25 - abs(obs[0]))*0.1
        stable_bonus = 1 if stable else -1
        reward = (
            dx * 1000                 
            + height
            - (self.checkFall()*1000)
            + stable_bonus 
            + centering_bonus
        )
        return float(reward)


    def get_observation(self):
        obs = []

        # Posição do robô em relação ao alvo
        dx, dy, dz = self.sim.getObjectPosition(self.robot, self.target)
        [roll,pitch,yaw] = self.sim.getObjectOrientation(self.robot, -1)
        obs.extend([dx, dy, dz])
        obs.extend([roll,pitch,yaw])

        # Velocidade linear e angular
        (vx, vy, vz), (wx, wy, wz) = self.sim.getObjectVelocity(self.robot)
        obs.extend([vx, vy, vz, wx, wy, wz])

        # Estados das juntas
        for jointName in self.joint_list:
            joint = self.handleDict[jointName]
            angle = self.sim.getJointPosition(joint)
            _, speed = self.sim.getObjectFloatParameter(joint, self.sim.jointfloatparam_velocity)
            obs.extend([angle, speed])

        # Direção do robô
        matrix = self.sim.getObjectMatrix(self.robot, -1)
        obs.append(matrix[0])  # direção em x

        return np.array(obs, dtype=np.float32)

    def check_done(self):
        return self.checkFall() or self.checkArrival() or not self.cg_inside()

    def checkFall(self):
        _, _, z = self.sim.getObjectPosition(self.robot, self.target)
        return z < 0.15

    def checkArrival(self):
        dx, _, _ = self.sim.getObjectPosition(self.robot, self.target)
        return abs(dx) < 1.0
    

    def cg_inside(self):
        points_dict = self.discover_base_polygon_points()
        edges = (
            (points_dict["RL"],points_dict["RR"]),
            (points_dict["RR"],points_dict["FR"]),
            (points_dict["FR"],points_dict["FL"]),
            (points_dict["FL"],points_dict["RL"]),
            )
        cnt = 0
        for edge in edges:
            (x1,y1),(x2,y2) = edge
            if (0<y1) != (0<y2) and 0 < x1 + ((0-y1)/(y2-y1))*(x2-x1):
                cnt += 1
        return cnt%2 == 1

    def discover_base_polygon_points(self):
        rl_matrix = self.get_joint_final_matrix("RL")
        rr_matrix = self.get_joint_final_matrix("RR")
        fl_matrix = self.get_joint_final_matrix("FL")
        fr_matrix = self.get_joint_final_matrix("FR")

        return {"RL":(rl_matrix[0][3],rl_matrix[1][3]),
                "RR":(rr_matrix[0][3],rr_matrix[1][3]),
                "FL":(fl_matrix[0][3],fl_matrix[1][3]),
                "FR":(fr_matrix[0][3],fr_matrix[1][3])}


    def get_joint_final_matrix(self,vertex):
        matrix = self.discover_foot_position(
            self.vertex_2_center[vertex]["x"],self.vertex_2_center[vertex]["y"],self.vertex_2_center[vertex]["z"],\
            self.upper_leg_lenght,self.lower_leg_lenght,\
            self.sim.getJointPosition(self.sim.getObject(f'/Doggy/{vertex}_upper_leg_joint')),self.sim.getJointPosition(self.sim.getObject(f'/Doggy/{vertex}_lower_leg_joint'))
            )
        return matrix

    def fill_robot_data(self):
        self.vertex_2_center = {}
        for vertex in ["RL","RR","FL","FR"]:
            vertex_2_center = self.sim.getObjectPosition(self.sim.getObject(f'/Doggy/{vertex}_upper_leg_joint'),self.sim.getObject('/Doggy/base_link_visual'))
            self.vertex_2_center[vertex] = {}
            self.vertex_2_center[vertex]["x"] = vertex_2_center[0]
            self.vertex_2_center[vertex]["y"] = vertex_2_center[1]
            self.vertex_2_center[vertex]["z"] = vertex_2_center[2]

        _, self.upper_leg_lenght, _ = self.sim.getObjectPosition(self.sim.getObject('/Doggy/RR_lower_leg_joint'),self.sim.getObject('/Doggy/RR_upper_leg_joint'))
        _, self.lower_leg_lenght, _ = self.sim.getObjectPosition(self.sim.getObject('/Doggy/RR_foot_joint'),self.sim.getObject('/Doggy/RR_lower_leg_joint'))



    def discover_foot_position(self,dx,dy,dz,upper_length,lower_length,theta_1,theta_2):
        roll, pitch, yaw = self.sim.getObjectOrientation(self.robot, -1)
        inv_roll = self.get_homgeneous_transform_matrix(-roll,0,0,0,"x")
        inv_pitch = self.get_homgeneous_transform_matrix(-pitch,0,0,0,"y")
        inv_yaw = self.get_homgeneous_transform_matrix(-yaw,0,0,0,"z")
        inverse_rotation_matrix = inv_roll @ inv_pitch @ inv_yaw
        r_0_1 = self.get_homgeneous_transform_matrix(theta_1,dx,dy,dz,"y")
        r_1_2 = self.get_homgeneous_transform_matrix(-math.pi/2,0,0,0,"x")
        r_2_3 = self.get_homgeneous_transform_matrix(theta_2,0,upper_length,0,"z")
        r_3_4 = self.get_homgeneous_transform_matrix(0,0,lower_length,0,"x")

        
        return inverse_rotation_matrix @ (((r_0_1 @ r_1_2) @ r_2_3) @ r_3_4)
    

    def get_homgeneous_transform_matrix(self,theta,dx,dy,dz,axis):
        c = np.cos(theta)
        s = np.sin(theta)
        if axis == "x":
            matrix = np.array([
                [1, 0, 0,dx],
                [0, c,-s,dy],
                [0, s, c,dz],
                [0, 0, 0, 1]])
        if axis == "y":
            matrix = np.array([
                [ c, 0, s,dx],
                [ 0, 1, 0,dy],
                [-s, 0, c,dz],
                [ 0, 0, 0, 1]])
        if axis == "z":
            matrix = np.array([
                [c,-s, 0,dx],
                [s, c, 0,dy],
                [0, 0, 1,dz],
                [0, 0, 0, 1]])
        return matrix
            


