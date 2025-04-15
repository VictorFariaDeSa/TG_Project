import gymnasium as gym
from gymnasium import spaces
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
from helper import plot
from collections import deque

MAX_STEPS = 1000

class Doggy_walker(gym.Env):
    metadata = {"render_modes": [], "render_fps": 60}

    def __init__(self):
        super().__init__()
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32)

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

        self.score = 0
        self.n_steps = 0

        self.sim.stopSimulation()
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            time.sleep(0.1)

        self.sim.setStepping(True)
        self.sim.startSimulation()

        self.last_x = self.sim.getObjectPosition(self.robot, self.target)[0]
        self.last_time = self.sim.getSimulationTime()

        plot(self.plot_scores, self.plot_mean_scores)

        obs = self.get_observation()
        info = {}

        return obs, info

    def getHandles(self):
        handleDict = {}
        for name in self.joint_list:
            handleDict[name] = self.sim.getObject(f"/Doggy/{name}")
        return handleDict

    def get_reward(self, action):
        obs = self.get_observation()

        dx = obs[0] - self.last_x
        height = obs[2]  # z
        self.last_x = obs[0]

        curr_time = self.sim.getSimulationTime()
        dt = curr_time - self.last_time
        self.last_time = curr_time

        # Penaliza esforço, recompensa avanço e altura
        effort = np.abs(action).sum() * dt
        reward = (dx * 10.0) - effort - (self.checkFall() * 1000.0) + height*10

        return float(reward)

    def get_observation(self):
        obs = []

        # Posição do robô em relação ao alvo
        dx, dy, dz = self.sim.getObjectPosition(self.robot, self.target)
        obs.extend([dx, dy, dz])

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
        return self.checkFall() or self.checkArrival()

    def checkFall(self):
        _, _, z = self.sim.getObjectPosition(self.robot, self.target)
        return z < 0.2

    def checkArrival(self):
        dx, _, _ = self.sim.getObjectPosition(self.robot, self.target)
        return abs(dx) < 1.0
