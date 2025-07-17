import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# 🎯 Red neuronal para la Q-Function
class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.output = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.output(x)

# 🧠 Agente DQL
class DQLAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        # 🛡️ Forzar reward a float escalar
        reward = float(np.squeeze(np.array(reward)))
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            act_values = self.model(state_tensor)
        return torch.argmax(act_values[0]).item()

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            # Convertir a tensores
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)

            current_q_values = self.model(state_tensor)
            current_q_value = current_q_values[0, action]

            # 🛡️ Forzar reward a escalar
            reward = float(reward)

            if done:
                target_q_value = torch.tensor(reward, dtype=torch.float32).to(self.device)
            else:
                with torch.no_grad():
                    next_q_values = self.model(next_state_tensor)
                    max_next_q_value = torch.max(next_q_values).item()
                    target_q_value = reward + self.gamma * max_next_q_value

                target_q_value = torch.tensor(target_q_value, dtype=torch.float32).to(self.device)

            # Calcular la pérdida
            loss = self.loss_fn(current_q_value, target_q_value)

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
