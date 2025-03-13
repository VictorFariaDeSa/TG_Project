from coppeliasim_zmqremoteapi_client import *
import time
from SimulationControl import createSimulation
from ppoAgent import Agent
import numpy as np
import torch
import sys
import signal
import matplotlib.pyplot as plt

def handle_interrupt(signum, frame):
    print("\n⚠️ Interrupção detectada! Salvando modelo...")
    agent.saveModel(model_path)
    plotScores(scores)
    save_scores(scores,memory_score_path)
    print("✅ Modelo salvo com sucesso!")
    while env.sim.getSimulationState() != env.sim.simulation_stopped:
        time.sleep(0.1)
    env.sim.stopSimulation()
    sys.exit(0)
    

def plotScores(scores):
    fig, ax = plt.subplots(figsize=(20,10))
    ax.plot(scores)
    ax.set_title("Scores Over Time")
    ax.set_xlabel("Episódio")
    ax.set_ylabel("Score")
    ax.grid()
    plt.show()

def save_scores(scores, filename):
    try:
        existing_scores = np.load(filename).tolist()
    except FileNotFoundError:
        existing_scores = []

    existing_scores.append(scores)
    np.save(filename, existing_scores) 


simulation_name = "Deep_Q_learning_mark02_100.100.neurons_225h"
model_path = f"models/{simulation_name}.pth"
memory_score_path = f"scores/{simulation_name}.npy"

if __name__ == "__main__":
    agent:Agent = Agent(
        gamma=0.99,
        epsilon=0.1,
        batch_size=250,
        n_actions=24,
        eps_end=0.1,
        input_dims=21,
        lr=1e-4)

    signal.signal(signal.SIGINT, handle_interrupt)
    n_games:int = 3000  
    scores: list[float] = []
    eps_history: list[float] = np.zeros(n_games)
    agent.LoadModel(path = model_path)

    env = createSimulation(agent.Q_eval.device)
    observation:torch.Tensor = env.reset()
    
    
    for i in range(n_games):
        start_time = time.time()
        score:int = 0
        done:bool = False
        while True:
            action = agent.policy(observation)
            observation_, reward, terminated = env.step(action)
            agent.store_transitions(observation,action,reward,observation_,done)
            agent.learn()
            score += reward
            observation = observation_
            if terminated:
                observation =  env.reset()
                elapsed_time = time.time()-start_time
                break
        scores.append(score)
        eps_history[i] = agent.epsilon

        avg_score:float = np.mean(scores[-100:])
    
        print(f"Episode {i} - Score: {score} - Avg_Score: {avg_score} - Epsilon: {agent.epsilon} - Tempo: {elapsed_time:.3f} s")
    agent.saveModel(model_path)
    plotScores(scores)
    save_scores(scores,memory_score_path)