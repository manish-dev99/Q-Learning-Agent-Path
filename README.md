# Q-Learning-Agent-Path
# 🤖 Animated Q-Learning Gridworld

Interactive **Q-Learning Gridworld** project built with **Streamlit**, where an agent learns to navigate a grid with obstacles and reach a goal. The agent’s path is **animated step by step**, making it visually engaging and ideal for learning reinforcement learning concepts.

[🔗 Live Demo](https://q-learning-agent-path-7.streamlit.app/)

---

## Features

- ✅ Interactive sidebar for **customizing the environment**:
  - Grid size, start & goal positions
  - Number and positions of obstacles
  - Q-Learning hyperparameters (α, γ, ε, episodes)
  - Animation speed

- ✅ Animated agent movement showing **step-by-step learning path**.

- ✅ Compact visualization suitable for **portfolio demonstrations**.

- ✅ Fully **cloud deployable** with Streamlit Cloud.

---

## How It Works

1. The agent starts at a **customizable start position**.
2. It uses **Q-Learning** to learn the optimal path to the goal while avoiding obstacles.
3. After training, the agent **follows the learned path**.
4. The app animates each step using **Matplotlib**, making it easy to visualize the agent’s decision-making.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/manish-dev99/animated-qlearning-gridworld.git
cd animated-qlearning-gridworld
