# TG_Project: Deep Reinforcement Learning Application to Teach a 4-Legged Robot How to Walk

## Introduction

This repository contains all files related to my graduation thesis, which focuses on optimizing a neural network to improve the walking process of a quadruped robot using deep reinforcement learning (DRL). The goal is to enable the robot to learn efficient locomotion strategies through trial and error using reinforcement learning techniques.

## Requirements

Before running the simulation code, ensure that you have the following dependencies installed:

- **CoppeliaSim** (for robot simulation)
- **Python 3.7+**
- **PyTorch** (for deep learning models)
- Additional Python libraries (listed in `requirements.txt`)

To install all necessary dependencies, run:
```bash
pip install -r requirements.txt
```

## Running CoppeliaSim

1. Locate the CoppeliaSim installation folder.
   
   **Example:** `~/Downloads/VREP/`
   
2. Start CoppeliaSim.
   
   **On Linux/Mac:**
   ```bash
   ~/Downloads/VREP/coppeliaSim.sh
   ```
   
3. Open the simulation scene.
   
   Currently, there are two main simulation scenes available:
   
   - **`simulation2.ttt`** → Upper and lower joints move from -90º to 90º.
   - **`simulation4.ttt`** → Upper joints move from -45º to 45º, and lower joints move from -90º to 90º.
   
   You can open the latest simulation version directly by running:
   ```bash
   ~/Downloads/VREP/coppeliaSim.sh tg_proj/simulations/<chosen_simulation>.ttt
   ```
   Alternatively, you can open CoppeliaSim manually and load the scene via:
   **File -> Open Scene -> `<chosen_simulation>.ttt`**
   
4. Run the agent code.
   
   Currently, there are two main agent implementations for the project:
   - **`Oldppo8nnAgent.py`**
   - **`DNQAgent.py`**
   
   Once CoppeliaSim is open, run the interaction script to establish a connection with CoppeliaSim and start data collection for training.

## Current Experiments

The current experiment evaluates the impact of modifying the return function from:
```python
returns = t_advantage + t_state_values
```
to:
```python
returns = t_rewards + gamma * critic_nn(t_states)
```

## Next Steps

The next set of planned experiments includes:

1. **Stop normalizing advantages** to observe the impact on training stability.
2. **Stop utilizing entropy** in the loss function to test convergence effects.
3. **Execute a learning action in each time step** to analyze training efficiency.
4. **Implement a deque memory** and stop resetting the memory after each episode.
5. **Normalize input data for neural networks** to improve stability and generalization.
6. **Create a critic network for each joint** to enhance control precision.
7. **Design a specific loss function for each actor-critic neural network pair** to optimize training.
8. **Reduce the number of layers** to evaluate performance with a simpler network structure.

## Contributions

Feel free to contribute to the project by submitting pull requests, reporting issues, or suggesting improvements. 

## License

This project is released under the MIT License.


