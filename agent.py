import torch
import random
import numpy as np
from collections import deque
from model import Linear_QNet,QTrainer
from helper import plot
from SimulationControl import createSimulation

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0.9
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(21,256,128,64,24)
        self.trainer = QTrainer(self.model,lr = LR,gamma=self.gamma)


    def get_state(self,env):
        return env.getObservation()

    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))

    def train_long_memory(self):
        if len(self.memory) >= BATCH_SIZE:
            mini_sample = random.sample(self.memory,BATCH_SIZE)
        else:
            mini_sample = self.memory

        states,actions,rewards,next_states,dones = zip(*mini_sample)
        
        self.trainer.train_step(states,actions,rewards,next_states,dones)
        

    def train_short_memory(self,state,action,reward,next_state,done):
        self.trainer.train_step(state,action,reward,next_state,done)

    def get_action(self,state):
        final_move = [0]*24
        if np.random.random() < self.epsilon and self.n_games%10 != 0:
            moves = []
            for idx in range(8):
                move = random.randint(0,2)
                moves.append(move+idx*3)
        else:
            state0 = torch.tensor(state,dtype=torch.float)
            prediction = self.model(state0)
            prediction_reshaped = prediction.view(-1, 3)
            moves = torch.argmax(prediction_reshaped, dim=1).tolist()
            moves = [move+(i*3) for i,move in enumerate(moves)]
        for move in moves:
            final_move[move] = 1
        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    plot_evaluation_scores = []
    total_score = 0
    record = float("-inf")
    agent = Agent()
    simulation = createSimulation("cpu")
    simulation.reset()
    while True:
        state_old = agent.get_state(simulation)
        final_move = agent.get_action(state_old)
        reward, done, score = simulation.step(final_move)
        state_new = agent.get_state(simulation)
        agent.train_short_memory(state=state_old,action=final_move,reward=reward,next_state=state_new,done=done)
        agent.remember(state_old,final_move,reward,state_new,done)
        
        if done:
            if agent.epsilon > 0.05:
                agent.epsilon -= 1e-4
            simulation.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()
            print(f"Game:{agent.n_games}   |Score:{score}     |Record:{record}  |Epsilon:{agent.epsilon}")
            plot_scores.append(score)
            total_score += score
            mean_score = total_score/agent.n_games
            plot_mean_scores.append(mean_score)
            if agent.n_games % 10 == 0:
                plot_evaluation_scores.append(score)
            plot(plot_scores,plot_mean_scores,plot_evaluation_scores)


if __name__ == "__main__":
    train()


