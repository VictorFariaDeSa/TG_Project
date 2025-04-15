import gymnasium as gym
import numpy as np
import torch
from customEnv_myNN import Agent
import torch.optim as optim
from helper import plot
from collections import deque

if __name__ == "__main__":
    env = gym.make('HalfCheetah-v5')
    model = Agent(
        gamma = 0.99,
        lam = 0.95,
        policy_clip = 0.2,
        input_size = env.observation_space.shape[0],
        inner_dimensions = [512, 256, 128],
        n_joints = env.action_space.shape[0],
        lr = 1e-4,
        epochs = 5,
        batch_size = 64,
        env = env
        )
    model.train()
