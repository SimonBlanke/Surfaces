"""DQN Reinforcement Learning on CartPole-v1."""

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from surfaces.modifiers import BaseModifier

from ..._base_rl import BaseReinforcementLearning


class DQNCartPoleFunction(BaseReinforcementLearning):
    """DQN Policy Optimization test function for CartPole-v1.

    **What is optimized:**
    This function optimizes the hyperparameters of a Deep Q-Network (DQN) agent
    learning to balance a pole on a cart. The search includes:
    - Hidden units in neural network (32, 64, 128, 256)
    - Learning rate (1e-4, 5e-4, 1e-3, 5e-3)
    - Gamma (discount factor: 0.95, 0.99, 0.995, 0.999)
    - Epsilon decay (exploration decay: 0.99, 0.995, 0.999)
    - Batch size for experience replay (32, 64, 128)

    **Reinforcement Learning concept:**
    RL agents learn by trial and error, receiving rewards for good actions.
    DQN uses a neural network to approximate Q-values (expected future rewards)
    for state-action pairs. The hyperparameters control learning speed,
    exploration vs exploitation, and memory usage.

    **What the score means:**
    The score is the mean episode reward over the last 20 episodes of training
    on the CartPole-v1 environment. In CartPole, the agent receives +1 reward
    per timestep the pole stays balanced, with episodes ending after 500 steps
    or when the pole falls. Higher scores (up to 500) indicate better policies
    that keep the pole balanced longer.

    **Optimization goal:**
    MAXIMIZE the mean episode reward. The goal is to find hyperparameters that
    enable the agent to:
    - Learn quickly (good learning rate, appropriate network size)
    - Balance exploration and exploitation (epsilon decay)
    - Look far enough ahead (gamma)
    - Learn stably from experience (batch size)

    A well-tuned DQN can achieve near-optimal performance (450-500 reward).

    **Computational cost:**
    Each evaluation trains a DQN agent for multiple episodes, making this
    moderately expensive (~20-60 seconds per evaluation). The default uses
    100 episodes for reasonable training with fast evaluation.

    Parameters
    ----------
    n_episodes : int, default=100
        Number of training episodes per evaluation.
    max_steps : int, default=500
        Maximum steps per episode (CartPole-v1 default).
    buffer_size : int, default=10000
        Size of experience replay buffer.
    n_jobs : int, default=2
        Number of CPU threads for PyTorch. Lower values reduce CPU load,
        keeping the system responsive during training.
    objective : str, default="maximize"
        Either "minimize" or "maximize".
    modifiers : list of BaseModifier, optional
        List of modifiers to apply to function evaluations.

    Examples
    --------
    >>> from surfaces.test_functions.machine_learning import DQNCartPoleFunction
    >>> func = DQNCartPoleFunction(n_episodes=50)
    >>> func.search_space
    {'hidden_units': [32, 64, 128, 256], 'learning_rate': [0.0001, 0.0005, ...], ...}
    >>> result = func({"hidden_units": 128, "learning_rate": 0.001,
    ...                "gamma": 0.99, "epsilon_decay": 0.995, "batch_size": 64})
    >>> print(f"Mean episode reward: {result:.2f}")

    Notes
    -----
    Requires gymnasium. Install with:
        pip install gymnasium

    The function uses a simplified DQN implementation focused on demonstrating
    RL hyperparameter optimization. For production RL training, use libraries
    like Stable-Baselines3 or RLlib.

    CartPole-v1 is a classic RL benchmark where an agent must balance a pole
    on a moving cart by applying left/right forces. It's fast to train and
    provides clear feedback on hyperparameter quality.
    """

    name = "DQN CartPole"
    _name_ = "dqn_cartpole"
    __name__ = "DQNCartPoleFunction"

    para_names = ["hidden_units", "learning_rate", "gamma", "epsilon_decay", "batch_size"]
    hidden_units_default = [32, 64, 128, 256]
    learning_rate_default = [1e-4, 5e-4, 1e-3, 5e-3]
    gamma_default = [0.95, 0.99, 0.995, 0.999]
    epsilon_decay_default = [0.99, 0.995, 0.999]
    batch_size_default = [32, 64, 128]

    def __init__(
        self,
        n_episodes: int = 100,
        max_steps: int = 500,
        buffer_size: int = 10000,
        n_jobs: int = 2,
        objective: str = "maximize",
        modifiers: Optional[List[BaseModifier]] = None,
        memory: bool = False,
        collect_data: bool = True,
        callbacks: Optional[Union[Callable, List[Callable]]] = None,
        catch_errors: Optional[Dict[type, float]] = None,
        use_surrogate: bool = False,
    ):
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.buffer_size = buffer_size
        self.n_jobs = n_jobs

        super().__init__(
            objective=objective,
            modifiers=modifiers,
            memory=memory,
            collect_data=collect_data,
            callbacks=callbacks,
            catch_errors=catch_errors,
            use_surrogate=use_surrogate,
        )

    @property
    def search_space(self) -> Dict[str, Any]:
        """Search space for DQN hyperparameter optimization."""
        return {
            "hidden_units": self.hidden_units_default,
            "learning_rate": self.learning_rate_default,
            "gamma": self.gamma_default,
            "epsilon_decay": self.epsilon_decay_default,
            "batch_size": self.batch_size_default,
        }

    def _create_objective_function(self) -> None:
        """Create objective function for DQN optimization."""
        import random
        from collections import deque

        import gymnasium as gym
        import torch
        import torch.nn as nn
        import torch.optim as optim

        # Limit CPU threads to keep system responsive
        torch.set_num_threads(self.n_jobs)

        n_episodes = self.n_episodes
        max_steps = self.max_steps
        buffer_size = self.buffer_size

        def objective_function(params: Dict[str, Any]) -> float:
            # Environment
            env = gym.make("CartPole-v1")
            state_size = env.observation_space.shape[0]
            action_size = env.action_space.n

            # Q-Network
            class QNetwork(nn.Module):
                def __init__(self, state_size, action_size, hidden_units):
                    super(QNetwork, self).__init__()
                    self.fc1 = nn.Linear(state_size, hidden_units)
                    self.fc2 = nn.Linear(hidden_units, hidden_units)
                    self.fc3 = nn.Linear(hidden_units, action_size)

                def forward(self, x):
                    x = torch.relu(self.fc1(x))
                    x = torch.relu(self.fc2(x))
                    return self.fc3(x)

            # Initialize network and optimizer
            q_network = QNetwork(state_size, action_size, params["hidden_units"])
            optimizer = optim.Adam(q_network.parameters(), lr=params["learning_rate"])
            criterion = nn.MSELoss()

            # Experience replay buffer
            replay_buffer = deque(maxlen=buffer_size)

            # Training parameters
            gamma = params["gamma"]
            epsilon = 1.0
            epsilon_decay = params["epsilon_decay"]
            epsilon_min = 0.01
            batch_size = params["batch_size"]

            # Track rewards
            episode_rewards = []

            # Training loop
            for episode in range(n_episodes):
                state, _ = env.reset()
                total_reward = 0

                for step in range(max_steps):
                    # Epsilon-greedy action selection
                    if random.random() < epsilon:
                        action = env.action_space.sample()
                    else:
                        with torch.no_grad():
                            state_tensor = torch.FloatTensor(state).unsqueeze(0)
                            q_values = q_network(state_tensor)
                            action = q_values.argmax().item()

                    # Take action
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    total_reward += reward

                    # Store experience
                    replay_buffer.append((state, action, reward, next_state, done))

                    # Train on batch
                    if len(replay_buffer) >= batch_size:
                        batch = random.sample(replay_buffer, batch_size)
                        states, actions, rewards, next_states, dones = zip(*batch)

                        states = torch.FloatTensor(states)
                        actions = torch.LongTensor(actions)
                        rewards = torch.FloatTensor(rewards)
                        next_states = torch.FloatTensor(next_states)
                        dones = torch.FloatTensor(dones)

                        # Current Q values
                        current_q = q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

                        # Target Q values
                        with torch.no_grad():
                            next_q = q_network(next_states).max(1)[0]
                            target_q = rewards + gamma * next_q * (1 - dones)

                        # Update network
                        loss = criterion(current_q, target_q)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    state = next_state

                    if done:
                        break

                # Decay epsilon
                epsilon = max(epsilon_min, epsilon * epsilon_decay)

                # Track reward
                episode_rewards.append(total_reward)

            env.close()

            # Return mean reward of last 20 episodes
            if len(episode_rewards) >= 20:
                mean_reward = np.mean(episode_rewards[-20:])
            else:
                mean_reward = np.mean(episode_rewards)

            return mean_reward

        self.pure_objective_function = objective_function
