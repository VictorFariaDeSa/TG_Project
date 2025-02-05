from coppeliasim_zmqremoteapi_client import *
import torch
import math
import numpy as np

def startSimulation(jointList: list[str]):
    client = RemoteAPIClient()
    sim = client.require('sim')
    sim.setStepping(True)
    sim.startSimulation()
    robot = sim.getObject('/Doggy/')
    target = sim.getObject('/Target')
    handleDict = getHandles(sim,jointList)
 
    return client,sim,robot,target,handleDict

def getHandles(sim,jointNames: list[str]):
    handleDict: dict[str,int] = {}
    for name in jointNames:
        path = f"/Doggy/{name}"
        handleDict[name] = sim.getObject(path)
    return handleDict

def getEnviromentVector(sim, robot, target, jointHandler,jointList):
    env_vector = []
    dx, dy, dz = sim.getObjectPosition(robot, target)
    env_vector.extend([dx, dy, dz])
    
    for jointName in jointList:
        joint = jointHandler[jointName]
        joint_angle = sim.getJointPosition(joint)
        error, joint_speed = sim.getObjectFloatParameter(joint, sim.jointfloatparam_velocity)
        env_vector.extend([joint_angle, joint_speed])
    obj_matrix = sim.getObjectMatrix(robot, -1)
    correct_direction = obj_matrix[0]
    upsideDown = obj_matrix[10]
    env_vector.extend([correct_direction,upsideDown])
    
    return torch.tensor(env_vector)

def translateAction(sim, action:int,joint):
    match action:
        case 0:
            sim.setJointTargetVelocity(joint, -math.pi)
        case 1:
            sim.setJointTargetVelocity(joint, 0)
        case 2:
            sim.setJointTargetVelocity(joint, math.pi)
        case _:
            raise ValueError(f"Numero passado {action} para translate action não corresponde as ações")

def getReward(sim,robot,target):
    dx, dy, dz = sim.getObjectPosition(robot, target)
    upsideBonus = 0 if checkUpsideDown(sim,robot) else -10
    reach_bonus = 100 if dx < 5 else 0
    return -1
    return 30-dx+upsideBonus-dy+reach_bonus

def checkFall(sim,robot,target):
    dx, dy, dz = sim.getObjectPosition(robot, target)
    return abs(dz) > 10

def checkUpsideDown(sim,robot):
        obj_matrix = sim.getObjectMatrix(robot, -1)
        if obj_matrix[10] > 0:
            return 0
        else:
            return 1

def checkDirection(sim,robot):
        obj_matrix = sim.getObjectMatrix(robot, -1)
        if obj_matrix[0] < 0:
            return 1
        else:
            return 0


def checkArrival(sim,robot,target):
    dx, dy, dz = sim.getObjectPosition(robot, target)
    return dx < 1

def checkDone(sim,robot,target):
     return (checkFall(sim = sim,robot = robot,target = target) or checkArrival(sim = sim,robot = robot,target = target))

def takeActions(sim,robot,target,actions,jointHandler,jointList):
    for joint_index, jointName in enumerate(jointList):
            joint = jointHandler[jointName]
            translateAction(sim = sim, action = actions[joint_index]-3*joint_index, joint = joint)
    sim.step()
    env_vector = getEnviromentVector(sim = sim, robot = robot, target = target, jointHandler = jointHandler,jointList = jointList)
    reward = getReward(sim = sim,robot = robot,target = target)
    done = checkDone(sim = sim, robot = robot, target = target)
    return env_vector,reward,done
            
            