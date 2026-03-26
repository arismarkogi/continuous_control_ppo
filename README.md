# Proximal Policy Optimization (PPO) - Continuous Control

[![Udacity - Deep Reinforcement Learning Nanodegree](https://img.shields.io/badge/Udacity-Deep%20RL-blue.svg)](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)

This repository contains a PyTorch implementation of **Proximal Policy Optimization (PPO)** used to solve the multi-agent Unity Reacher environment. This project is part of the Udacity Deep Reinforcement Learning Nanodegree.

## The Environment

In this environment, 20 double-jointed robotic arms must learn to move their "hands" into moving target spheres and track them. 

* **State Space:** `33` dimensions (position, rotation, velocity, and angular velocities of the arm).
* **Action Space:** `4` continuous dimensions (torques applied to two joints). Every entry in the action vector is a number between `-1` and `1`.
* **Reward:** `+0.1` for each step the agent's hand is inside the goal location.
* **Goal:** The environment is considered solved when the **average score across all 20 agents is +30.0 or higher over 100 consecutive episodes.**

## The Algorithm: Why PPO?

While Deep Deterministic Policy Gradient (DDPG) is a standard choice for continuous control, it can be highly unstable and sensitive to hyperparameters. This project implements an **Actor-Critic PPO** agent, featuring:
* **Stochastic Continuous Policy:** The Actor outputs a parameterized Gaussian distribution (Mean and learnable Standard Deviation).
* **Generalized Advantage Estimation (GAE):** For stable, low-variance value targets.
* **Clipped Surrogate Objective:** Ensures the policy does not suffer from catastrophic forgetting by limiting the size of gradient updates.
* **Orthogonal Initialization:** To prevent vanishing/exploding gradients early in training.

## Project Structure
* `model.py`: PyTorch neural network architectures for the PPO Actor and Critic.
* `memory.py`: The Rollout Buffer used to store on-policy trajectories.
* `agent.py`: The PPO agent class containing the GAE math and clipping update logic.
* `train.py`: The main training loop that interacts with the Unity environment.
* `checkpoint_actor.pth` / `checkpoint_critic.pth`: Saved model weights of the successful agent.
* `report.pdf`: A detailed mathematical write-up of the methodology.
