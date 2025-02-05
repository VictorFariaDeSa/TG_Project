from coppeliasim_zmqremoteapi_client import *
import time
from SimulationControl import startSimulation, getEnviromentVector, takeActions
from neuralNetworkModel import Agent
import numpy as np
import torch


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
    
    n_games:int = 500
    scores: list[float] = np.zeros(n_games)
    eps_history: list[float] = np.zeros(n_games)

    for i in range(n_games):
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
        while sim.getSimulationState() != sim.simulation_stopped:
            time.sleep(0.1)
        print("Simulation successfully ended")
        scores[i] = score
        eps_history[i] = agent.epsilon

        avg_score:float = np.mean(scores[-100:])

        print(f"Episode {i} - Score: {score} - Avg_Score: {avg_score} - Epsilon: {agent.epsilon}")