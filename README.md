# Reinforcement Learning: Frozen Lake

## Overview
This repository contains implementations of both Q-learning (QL) and Deep Q-learning (DeepQL) to solve the Frozen Lake environment from OpenAI's Gym library. 

### Environment: Frozen Lake

Frozen Lake is a grid-world environment where an agent must navigate from a start position to a goal without falling into holes. The lake is made up of four types of blocks: Start (S), Goal (G), Frozen (F), and Hole (H). The agent must reach the Goal (G) from the Start (S) without falling into a Hole (H). The catch is that the Frozen (F) blocks are slippery, so the agent won't always move in the intended direction.

![Frozen Lake](https://github.com/VenturaHaze/ReinforcementLearning/blob/master/frozen_lake.gif)

Detailed description of the Frozen Lake environment can be found [here](https://www.gymlibrary.dev/environments/toy_text/frozen_lake/).

## Code Structure

### Files in the Repository:

1. **frozen_lake_map.pkl**: Pre-generated map for the Frozen Lake environment.
2. **generate_map.py**: Script to create and save the map for the Frozen Lake environment.
3. **model.pth**: Saved model weights for the Deep Q-learning neural network.
4. **model.py**: Contains the neural network architecture for Deep Q-learning.
5. **qtable.pkl**: Saved Q-table from Q-learning.
6. **simulation_QL.py**: Script to simulate the agent's behavior on Frozen Lake using Q-learning.
7. **simulation_deepQL.py**: Script to simulate the agent's behavior on Frozen Lake using Deep Q-learning.
8. **train_QL.py**: Script to train the agent on the Frozen Lake environment using Q-learning.
9. **train_deepQL.py**: Script to train the agent on the Frozen Lake environment using Deep Q-learning.
10. **utils.py**: Contains utility functions and classes, such as the `PolynomialEpsilonDecay` for epsilon decay during training.

### How to Run:

1. **Generate the map**:  
    ```bash
    python generate_map.py
    ```

2. **Train using Q-learning**:  
    ```bash
    python train_QL.py --episodes 1000
    ```

3. **Simulate using Q-learning**:  
    ```bash
    python simulation_QL.py --episodes 100
    ```

4. **Train using Deep Q-learning**:  
    ```bash
    python train_deepQL.py --episodes 1000
    ```

5. **Simulate using Deep Q-learning**:  
    ```bash
    python simulation_deepQL.py --episodes 100
    ```
