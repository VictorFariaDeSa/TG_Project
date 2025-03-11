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
        actions =  torch.tensor(actions,dtype=int)
        actions_reshaped = actions.view(-1, 3)
        for joint_index, jointName in enumerate(self.jointList):
                joint = self.jointHandler[jointName]
                actions_subset = actions_reshaped[joint_index]
                self.translateAction(action = torch.argmax(actions_subset).item(), joint = joint)
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
                self.sim.setJointTargetVelocity(joint, -math.pi/5)
            case 1:
                self.sim.setJointTargetVelocity(joint, 0)
            case 2:
                self.sim.setJointTargetVelocity(joint, math.pi/5)
            case _:
                raise ValueError(f"Numero passado {action} para translate action não corresponde as ações")
             
    def getObservation(self):
        env_vector = []
        dx, dy, dz = self.sim.getObjectPosition(self.robot, self.target)
        env_vector.extend([dx, dy, dz])
        
        for jointName in self.jointList:
            joint = self.jointHandler[jointName]
            joint_angle = self.sim.getJointPosition(joint)
            error, joint_speed = self.sim.getObjectFloatParameter(joint, self.sim.jointfloatparam_velocity)
            env_vector.extend([joint_angle, joint_speed])
        obj_matrix = self.sim.getObjectMatrix(self.robot, -1)
        correct_direction = obj_matrix[0]
        upsideDown = obj_matrix[10]
        env_vector.extend([correct_direction,upsideDown])
        
        return np.array(env_vector)
    
    def getReward(self):
        dx, dy, dz = self.sim.getObjectPosition(self.robot, self.target)
        upsideBonus = 0 if self.checkUpsideDown() else -10
        reach_bonus = 100 if dx < 5 else 0
        belly_bonus = self.checkBellyTouchingGround()
        return -dx-(5*belly_bonus) 
        return 30-dx+upsideBonus-dy+reach_bonus
    
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
