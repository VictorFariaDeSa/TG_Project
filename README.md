# TG_Project: Deep Reinforcement Learning Application to Teach a 4-Legged Robot How to Walk

## Introduction

This repository contains all the files related to my graduation thesis, which focuses on tuning a neural network to optimize the walking process of a quadruped robot using deep reinforcement learning.

## Requirements

Before running the simulation code, ensure that the following requirements are installed:

- CoppeliaSim
- Python 3.7+
- PyTorch
- Additional Python libraries

To install all necessary dependencies, run:

- pip install -r requirements.txt

## How to Run CoppeliaSim

1. Locate the CoppeliaSim installation folder
   Example: ~/Downloads/VREP/
2. Start CoppeliaSim

   **On Linux/Mac:**
~/Downloads/VREP/coppeliaSim.sh

3. Open the simulation scene

You can open the last simulation version directly by running:

~/Downloads/VREP/coppeliaSim.sh tg_proj/simulations/simulation2.ttt

Alternatively, open CoppeliaSim and manually load simulation2.ttt from File -> Open Scene -> simulation2.ttt.

4. Run the mark02.py code

Once CoppeliaSim is open, run the interaction script:

python mark02.py

This script connects to CoppeliaSim and starts data collection for training.




