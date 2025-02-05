from coppeliasim_zmqremoteapi_client import *
import time
import math
from SimulationControl import startSimulation, getEnviromentVector, translateAction, getReward, checkFall, checkDirection
from neuralNetworkModel import Policy,loadModel,saveModel
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import signal
import sys

def handle_interrupt(signum, frame):
    print("\n⚠️ Interrupção detectada! Salvando modelo...")
    saveModel(nn_model, model_path)
    print("✅ Modelo salvo com sucesso!")
    while sim.getSimulationState() != sim.simulation_stopped:
        time.sleep(0.1)
    sim.stopSimulation()
    sys.exit(0)
    
    
model_path = "models/model_weights_L1.pth"
nn_model = loadModel(model_path)

optimizer = optim.SGD(nn_model.parameters(), lr=0.01)
criterion = nn.L1Loss()
policy = Policy()

signal.signal(signal.SIGINT, handle_interrupt)

start_time = time.time()
while True:
    client, sim, robot, target, jointHandler = startSimulation()
    sim.setInt32Parameter(sim.intparam_speedmodifier,2)
    while sim.getSimulationTime() < 50 and not checkFall(sim,robot,target):
        
        env_vector = getEnviromentVector(sim = sim, robot = robot, target = target, jointHandler = jointHandler)

        results = nn_model(env_vector)

        np_array = results.detach().numpy() 
        target_data = results.clone().detach().numpy()
        actions = []
        for joint_index, joint in enumerate(jointHandler.values()):
            node_referenced = np_array[joint_index * 3 : joint_index * 3 + 3]
            action = policy.e_greedy_choose(node_referenced,0.1)
            translateAction(sim = sim, action = action, joint = joint)
            actions.append(action)
        sim.step()

        indices = [action + 3 * index for action, index in enumerate(actions)]
        target_data[indices] = getReward(sim = sim, robot = robot, target = target)

        target_tensor = torch.tensor(target_data, dtype=torch.float32)

        loss = criterion(results, target_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    sim.stopSimulation()
    while sim.getSimulationState() != sim.simulation_stopped:
        time.sleep(0.1)
    print("Simulation successfully ended")
saveModel(nn_model,model_path)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tempo decorrido: {elapsed_time:.5f} segundos")






