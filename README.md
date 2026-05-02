# Safe Reinforcement Learning Benchmarks

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Tested-EE4C2C.svg)](https://pytorch.org/)
[![OmniSafe](https://img.shields.io/badge/OmniSafe-Supported-brightgreen.svg)](https://github.com/PKU-Alignment/omnisafe)

This repository contains the codebase for evaluating constrained policy optimization methods across structurally diverse environments. We benchmark **PPO, PPO-Lag, CPO, and LTL-PPO** on both a high-dimensional continuous physics engine and a custom discrete hierarchical gridworld.

**Authors:** Arka Ian Goswami & Reza Alvandi

---

## Overview

Standard reinforcement learning (RL) strictly maximizes cumulative return, which can lead to unsafe exploitation of environments in risk-sensitive scenarios. This project evaluates how different Safe RL algorithms enforce safety constraints and how their reliability depends on the underlying task structure.

### Key Findings
1. **Continuous Control:** **PPO-Lag** successfully balances the performance-safety trade-off in high-dimensional spaces. Conversely, strict analytic methods like **CPO** exhibit high variance and remain vulnerable to approximation errors, leading to catastrophic policy collapse.
2. **Discrete Hierarchical Tasks:** Standard policy gradient methods are highly fragile to reward design. When hierarchical dependencies are subjected to sparse or penalized reward structures (e.g., an early-goal penalty), PPO and PPO-Lag fail to bootstrap learning. **LTL-PPO** overcomes this by recovering the necessary inductive bias directly from formal Linear Temporal Logic specifications.

---

## Environments

### 1. Continuous Control: `SafetyPointGoal1-v0`
* **Framework:** [OmniSafe](https://github.com/PKU-Alignment/omnisafe)
* **Description:** A point-mass agent must navigate to randomized goal locations while avoiding dynamically placed hazards.
* **Constraint:** Strict safety cost limit ($d = 25.0$).

### 2. Discrete Gridworld: `SafeTreasureGoal`
* **Description:** A custom Markov Decision Process (MDP) defined on a 2D lattice. The agent must satisfy a strict temporal ordering: `Collect Key -> Open Door -> Reach Goal`, while avoiding cliffs and walls.
* **Variations:** Evaluated under standard sparse rewards and an **early-goal penalty** framework to test algorithmic resilience to reward shaping.

---

## Algorithms Implemented

*   **PPO (Proximal Policy Optimization):** Acts as the unconstrained baseline.
*   **PPO-Lag (PPO-Lagrangian):** Utilizes a dynamic Lagrangian multiplier (dual ascent) to adaptively penalize constraint violations.
*   **CPO (Constrained Policy Optimization):** Solves an analytical trusted-region optimization problem to strictly enforce KL-divergence bounds and cost constraints simultaneously.
*   **LTL-PPO:** Integrates Linear Temporal Logic (LTL) to provide inductive bias for hierarchical discrete tasks.

---

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/reza2002801/safe-rl-benchmarks.git](https://github.com/reza2002801/safe-rl-benchmarks.git)
   cd safe-rl-benchmarks