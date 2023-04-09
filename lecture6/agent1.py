import numpy as np
import math

class E_SARSA:
    def __init__(self, states_n, actions_n, alpha, gamma, epsilon):
        self.states_n = states_n
        self.actions_n = actions_n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.reset()

    def reset(self):
        self.episode = 0
        self.step = 0
        self.state = 0
        self.action = 0
        self.next_state = 0
        self.next_action = 0
        self.reward = 0
        self.done = False
        self.q_table = np.zeros((self.states_n, self.actions_n))

    def update(self, state, action, next_state, next_action, reward, terminated, truncated):
        self._update(state, action, next_state, next_action, reward, terminated, truncated)
        
        # valor esperado para el siguiente estado <> 
        expected_value = np.sum(self.policy(next_state) * self.q_table[next_state])

        td_error = reward + self.gamma * expected_value - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

    def _update(self, state, action, next_state, next_action, reward, terminated, truncated):
        if self.done:
            self.step = 0
            self.done = False

        self.step += 1
        self.state = state
        self.action = action
        self.next_state = next_state
        self.next_action = next_action
        self.reward = reward

        if terminated or truncated:
            self.episode += 1
            self.done = True

    def get_action(self, state, mode):
        if mode == "random":
            return np.random.choice(self.actions_n)
        elif mode == "greedy":
            return np.argmax(self.q_table[state])
        elif mode == "epsilon-greedy":
            if np.random.uniform(0, 1) < self.epsilon:
                return np.random.choice(self.actions_n)
            else:
                return np.argmax(self.q_table[state])

    def policy(self, state):
        action_probs = np.ones(self.actions_n) * self.epsilon / self.actions_n
        best_action = np.argmax(self.q_table[state])
        action_probs[best_action] += 1 - self.epsilon
        return action_probs

    def render(self, mode="step"):
        if mode == "step":
            print(
                f"Episode: {self.episode}, Step: {self.step}, State: {self.state}, Action: {self.action}, ",
                end="",
            )
            print(
                f"Next state: {self.next_state}, Reward: {self.reward}"
            )
        elif mode == "values":
            print(f"Q-Table: {self.q_table}")