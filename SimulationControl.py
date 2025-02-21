from coppeliasim_zmqremoteapi_client import *
import torch
import math
import numpy as np
import time

def createSimulation(num_robots,device):
    client = RemoteAPIClient()
    sim = client.require('sim')
    robot_list = []
    for i in range(num_robots):
        robot = Doggy_robot(sim,i,device)
        robot_list.append(robot)
    return env(client,sim,robot_list,device)

class Doggy_robot():
    def __init__(self,sim,i,device):
        self.i = i
        self.path = f"/Doggy[{i}]"
        self.sim = sim
        self.variable = sim.getObject(self.path)
        self.target = sim.getObject(f"/Target[{i}]")
        self.device = device
        self.terminated = False
        self.joint_list = ["RR_upper_leg_joint",
        "RL_upper_leg_joint",
        "FR_upper_leg_joint",
        "FL_upper_leg_joint",
        "RR_lower_leg_joint",
        "RL_lower_leg_joint",
        "FR_lower_leg_joint",
        "FL_lower_leg_joint"]
        self.joint_handler = self.getHandles(self.joint_list)

    def getHandles(self,jointNames: list[str]):
        handleDict: dict[str,int] = {}
        for name in jointNames:
            path = f"/Doggy[{self.i}]/{name}"
            handleDict[name] = self.sim.getObject(path)
        return handleDict
    
    def checkFall(self):
        dx, dy, dz = self.sim.getObjectPosition(self.variable, self.target)
        return abs(dz) > 10

    def checkArrival(self):
        dx, dy, dz = self.sim.getObjectPosition(self.variable, self.target)
        return dx < 1

    def checkDone(self):
        return (self.checkFall() or self.checkArrival())
    
    def updateTerminated(self):
        self.terminated = self.checkDone()
    
    def getObservation(self):
        env_vector = []
        dx, dy, dz = self.sim.getObjectPosition(self.variable, self.target)
        env_vector.extend([dx, dy, dz])

        for jointName in self.joint_list:
            joint = self.joint_handler[jointName]
            joint_angle = self.sim.getJointPosition(joint)
            error, joint_speed = self.sim.getObjectFloatParameter(joint, self.sim.jointfloatparam_velocity)
            env_vector.extend([joint_angle, joint_speed])

        obj_matrix = self.sim.getObjectMatrix(self.variable, -1)
        correct_direction = obj_matrix[0]
        upsideDown = obj_matrix[10]
        env_vector.extend([correct_direction, upsideDown])
        return torch.tensor(env_vector).to(self.device)

    def getReward(self):
            dx, dy, dz = self.sim.getObjectPosition(self.variable, self.target)
            upsideBonus = 0 if self.checkUpsideDown() else -10
            reach_bonus = 100 if dx < 5 else 0
            reward = -dx
            return reward
    
    def checkUpsideDown(self):
        obj_matrix = self.sim.getObjectMatrix(self.variable, -1)
        if obj_matrix[10] > 0:
            return 0
        else:
            return 1
    
    def checkDirection(self):
            obj_matrix = self.sim.getObjectMatrix(self.variable, -1)
            if obj_matrix[0] < 0:
                return 1
            else:
                return 0
    
    def translateSpecificAction(self, action:int,joint):
        match action:
            case 0:
                self.sim.setJointTargetVelocity(joint, -math.pi/5)
            case 1:
                self.sim.setJointTargetVelocity(joint, 0)
            case 2:
                self.sim.setJointTargetVelocity(joint, math.pi/5)
            case _:
                raise ValueError(f"Numero passado {action} para translate action não corresponde as ações")
    
    def translateActionVector(self,actions):
        for joint_index, jointName in enumerate(self.joint_list):
                joint = self.joint_handler[jointName]
                self.translateSpecificAction(action = actions[joint_index]-3*joint_index, joint = joint)
        return self.getObservation(),self.getReward(), self.checkDone()


class env():
    def __init__(self,client, sim, robot_list,device):
          self.client = client
          self.sim = sim
          self.robot_list = robot_list
          self.device = device

    def step(self,actions):
        if actions.shape[0] != len(self.robot_list):
            raise ValueError(f"Dimensões da matriz de ações esta incorreta {actions.shape[0]} deveria ser igual a {len(self.robot_list)}")
        for robot_index,robot in enumerate(self.robot_list):
            robot.translateActionVector(actions[robot_index])
        self.sim.step()
        return self.getCompleteObservation(),self.getCompleteReward(), self.getTerminalStates()

    def reset(self):
        self.sim.stopSimulation()
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            time.sleep(0.1)
        self.sim.setStepping(True)
        self.sim.startSimulation()
        return self.getCompleteObservation()
             
    def checkSimTime(self,time):
        return self.sim.getSimulationTime() >= time
        
    def getCompleteObservation(self):
        env_vectors = []
        for robot in self.robot_list:
            if not robot.terminated:
                env_vector = robot.getObservation()
                env_vectors.append(torch.tensor(env_vector, dtype=torch.float32))

        return torch.stack(env_vectors).to(self.device)
    
    def getCompleteReward(self):
        reward_column = []
        for robot in self.robot_list:
            if not robot.terminated:
                reward_column.append(robot.getReward())
        return torch.tensor(reward_column, dtype=torch.float32).unsqueeze(1).to(self.device)
    
    def checkSimTime(self,time):
        return self.sim.getSimulationTime() >= time

    def getTerminalStates(self):
        return torch.tensor([robot.checkDone() for robot in self.robot_list], dtype=torch.float32).unsqueeze(1).to(self.device)

    def update_robots_terminated(self):
        for robot in self.robot_list:
            if not robot.terminated:
                robot.updateTerminated()

    def checkDone(self):
        return all(robot.terminated for robot in self.robot_list) or self.checkSimTime(50)


