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
        dx, dy, dz = self.sim.getObjectPosition(self.robot, self.target)
        self.last_dx = dx
        self.last_dy = dy
        self.actions = None

    def step(self,actions):
        for joint_index, jointName in enumerate(self.jointList):
                joint = self.jointHandler[jointName]
                self.translateAction(action = actions[joint_index], joint = joint)
        self.sim.step()
        self.score+= self.getReward()
        return self.getReward(), self.checkDone(), self.score



    def C_step(self,actions):
        return self.getReward(), self.checkDone(), self.score


    def reset(self):
        self.score = 0
        self.sim.stopSimulation()
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            time.sleep(0.1)
        self.sim.setStepping(True)
        self.sim.startSimulation()
        dx, dy, dz = self.sim.getObjectPosition(self.robot, self.target)
        self.last_dx = dx
        return self.getObservation()
    
    def checkFall(self):
        dx, dy, dz = self.sim.getObjectPosition(self.robot, self.target)
        return abs(dz) > 10

    def checkArrival(self):
        dx, dy, dz = self.sim.getObjectPosition(self.robot, self.target)
        return abs(dx) < 1

    def checkSimTime(self,time):
        return self.sim.getSimulationTime() >= time
        
    def checkDone(self):
        return (self.checkFall() or self.checkArrival() or self.checkBellyTouchingGround() or self.checkSimTime(60))

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
        env_vector.extend([correct_direction])

        # env_vector.extend([correct_direction,upsideDown])
        
        return np.array(env_vector)
    

    def get_ok_height(self):
        dx, dy, dz = self.sim.getObjectPosition(self.robot, self.target)
        return dz > 0.3
    
    def check_vel_0(self):
        [vx, vy, vz], [wx, wy, wz] = self.sim.getObjectVelocity(self.robot)
        return vx < 0.1

    def getReward(self,k_distance = 250,k_energy = -0.1,k_speed = 5,k_vel_zero = -1,k_height = -10, k_angular_speed = -1, k_laydown = -1000,k_stable = 1, k_yOffset = -5, k_reach = 100000,k_effort = 0):
        
        height_goal = 0.35
        d_max = 25
        
        dx, dy, dz = self.sim.getObjectPosition(self.robot, self.target)

        dx_gain = self.last_dx-dx
        dy_gain = abs(self.last_dy)-abs(dy)

        self.last_dx = dx
        self.last_dy = dy

        # [vx, vy, vz], [wx, wy, wz] = self.sim.getObjectVelocity(self.robot)
        reach = self.checkArrival()
        laydown = self.checkBellyTouchingGround()
        fall = self.checkFall()
        pitch_angle = self.sim.getObjectOrientation(self.robot, -1)[1]
        roll_angle = self.sim.getObjectOrientation(self.robot, -1)[0]
        yaw_angle = self.sim.getObjectOrientation(self.robot, -1)[2] #para frente é 3.14
        vel_0 = self.check_vel_0()
        max_angle = math.pi/3
        stable = abs(yaw_angle)  < max_angle and abs(pitch_angle) < max_angle and abs(roll_angle) < max_angle
        

        reward = 0
        reward += k_distance * dx_gain
        reward += k_stable*stable
        reward += k_reach*reach

        reward += k_yOffset*dy_gain
        reward += k_energy*torch.sum(self.actions)
        reward += k_laydown*laydown

        return  reward  
    
    
    def checkBellyTouchingGround(self):
        dx, dy, dz = self.sim.getObjectPosition(self.robot, self.target)
        return dz < 0.25 #era 0.15

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
