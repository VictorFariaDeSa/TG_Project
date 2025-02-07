from coppeliasim_zmqremoteapi_client import *
import time
from SimulationControl import startSimulation, getEnviromentVector, takeActions
from neuralNetworkModel import Agent, loadModel,saveModel
import numpy as np
import torch
import sys
import signal

def handle_interrupt(signum, frame):
    print("\n⚠️ Interrupção detectada! Salvando modelo...")
    saveModel(agent.Q_eval, model_path)
    print("✅ Modelo salvo com sucesso!")
    while sim.getSimulationState() != sim.simulation_stopped:
        time.sleep(0.1)
    sim.stopSimulation()
    sys.exit(0)

model_path = "models/Deep_Q_learning_mark01.pth"

if __name__ == "__main__":
    agent:Agent = Agent(
        gamma=0.99,
        epsilon=1.0,
        batch_size=64,
        n_actions=24,
        eps_end=0.01,
        input_dims=21,
        lr=0.03)
    
    jointList = ["RR_upper_leg_joint",
        "RL_upper_leg_joint",
        "FR_upper_leg_joint",
        "FL_upper_leg_joint",
        "RR_lower_leg_joint",
        "RL_lower_leg_joint",
        "FR_lower_leg_joint",
        "FL_lower_leg_joint"]
    
    signal.signal(signal.SIGINT, handle_interrupt)
    n_games:int = 500
    scores: list[float] = []
    eps_history: list[float] = np.zeros(n_games)
    agent.LoadModel(path = model_path)

    for i in range(n_games):
        start_time = time.time()
        score:int = 0
        done:bool = False
        client, sim, robot, target, jointHandler= startSimulation(jointList=jointList)
        observation:torch.Tensor = getEnviromentVector(sim = sim,robot = robot,target = target,jointHandler = jointHandler,jointList = jointList)
        while not done and sim.getSimulationTime() < 50:
            nn_prediction:torch.Tensor = agent.Q_eval(observation = observation)
            actions:torch.Tensor = agent.choose_action(nn_prediction = nn_prediction)
            observation_,reward,done = takeActions(sim = sim,robot = robot,target = target,actions = actions,jointHandler = jointHandler,jointList = jointList)
            score += reward
            agent.store_transitions(state=observation,action=actions,reward=reward,state_=observation_,done=done)
            agent.learn()
            observation = observation_
        sim.stopSimulation()
        end_time = time.time()
        elapsed_time = end_time - start_time
        while sim.getSimulationState() != sim.simulation_stopped:
            time.sleep(0.1)
        print("Simulation successfully ended")
        scores.append(score)
        eps_history[i] = agent.epsilon

        avg_score:float = np.mean(scores[-100:])

        print(f"Episode {i} - Score: {score} - Avg_Score: {avg_score} - Epsilon: {agent.epsilon} - Tempo: {elapsed_time:.3f} s")
    agent.saveModel(model_path)