from coppeliasim_zmqremoteapi_client import *
import torch
import math
import numpy as np
import time

def createSimulation(device):
    jointList = ["RR_upper_leg_joint",
        "RL_upper_leg_joint",
        "FR_upper_leg_joint",
        "FL_upper_leg_joint",
        "RR_lower_leg_joint",
        "RL_lower_leg_joint",
        "FR_lower_leg_joint",
        "FL_lower_leg_joint"]
    client = RemoteAPIClient()
    sim = client.require('sim')
    robot = sim.getObject('/Doggy/')
    target = sim.getObject('/Target')
    handleDict = getHandles(sim,jointList)
 
    return env(client,sim,robot,target,handleDict,jointList,device)

def getHandles(sim,jointNames: list[str]):
    handleDict: dict[str,int] = {}
    for name in jointNames:
        path = f"/Doggy/{name}"
        handleDict[name] = sim.getObject(path)
    return handleDict
            
class env():
    def __init__(self,client, sim, robot, target, jointHandler,jointList,device):
        self.client = client
        self.sim = sim
        self.robot = robot
        self.target = target
        self.jointHandler = jointHandler
        self.jointList = jointList
        self.device = device
        self.score = 0

    def step(self,actions):
        for joint_index, jointName in enumerate(self.jointList):
                joint = self.jointHandler[jointName]
                self.translateAction(action = actions[joint_index], joint = joint)
        self.sim.step()
        self.score+= self.getReward()
        return self.getReward(), self.checkDone(), self.score

    def reset(self):
        self.score = 0
        self.sim.stopSimulation()
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            time.sleep(0.1)
        self.sim.setStepping(True)
        self.sim.startSimulation()
        return self.getObservation()
    
    def checkFall(self):
        dx, dy, dz = self.sim.getObjectPosition(self.robot, self.target)
        return abs(dz) > 10

    def checkArrival(self):
        dx, dy, dz = self.sim.getObjectPosition(self.robot, self.target)
        return dx < 1

    def checkSimTime(self,time):
        return self.sim.getSimulationTime() >= time
        
    def checkDone(self):
        return (self.checkFall() or self.checkArrival() or self.checkSimTime(25))

    def translateAction(self, action:int,joint):
        match action:
            case 0:
                self.sim.setJointTargetVelocity(joint, -math.pi/3)
            case 1:
                self.sim.setJointTargetVelocity(joint, 0)
            case 2:
                self.sim.setJointTargetVelocity(joint, math.pi/3)
            case _:
                raise ValueError(f"Numero passado {action} para translate action não corresponde as ações")
             
    def getObservation(self):
        env_vector = []
        dx, dy, dz = self.sim.getObjectPosition(self.robot, self.target)
        env_vector.extend([dx, dy, dz])
        velocity = self.sim.getObjectVelocity(self.robot)
        [vx, vy, vz], [wx, wy, wz] = velocity
        env_vector.extend([vx,vy,vz])
        for jointName in self.jointList:
            joint = self.jointHandler[jointName]
            joint_angle = self.sim.getJointPosition(joint)
            error, joint_speed = self.sim.getObjectFloatParameter(joint, self.sim.jointfloatparam_velocity)
            env_vector.extend([joint_angle, joint_speed])
        obj_matrix = self.sim.getObjectMatrix(self.robot, -1)
        correct_direction = obj_matrix[0]
        upsideDown = obj_matrix[10]
        # env_vector.extend([correct_direction,upsideDown])
        
        return np.array(env_vector)
    
    def getReward(self):
        dx, dy, dz = self.sim.getObjectPosition(self.robot, self.target)
        upside_penalty = -10 if self.checkUpsideDown() else 0
        reach_bonus = 100 if dx < 1 else 0  # Big bonus for reaching target
        progress_reward = -dx  # Reward for moving toward target
        belly_penalty = -5 if self.checkBellyTouchingGround() else 0
        height_penalty = -5 if dz < 0.15 else 0  # Penalize being too low
        
        # Combined reward to encourage:
        # 1. Moving toward target (progress_reward)
        # 2. Staying upright (upside_penalty)
        # 3. Not dragging belly (belly_penalty)
        # 4. Maintaining height (height_penalty)
        # 5. Reaching target (reach_bonus)
        return dz
        return progress_reward + upside_penalty + belly_penalty + height_penalty + reach_bonus
    
    def checkBellyTouchingGround(self):
        dx, dy, dz = self.sim.getObjectPosition(self.robot, self.target)
        return dz < 0.15

    def checkUpsideDown(self):
        obj_matrix = self.sim.getObjectMatrix(self.robot, -1)
        if obj_matrix[10] > 0:
            return 0
        else:
            return 1
        
    
    def checkDirection(self):
            obj_matrix = self.sim.getObjectMatrix(self.robot, -1)
            if obj_matrix[0] < 0:
                return 1
            else:
                return 0
