import gym
from keras.models import Sequential
from keras import layers 
from keras.optimizers import Adam
from collections import deque
import random
import numpy as np
import matplotlib.pyplot as plt

# ENVIRONMENT
env = gym.make("LunarLander-v2", render_mode="human")

# AGENT
class DQNAgent:
    def __init__(self):
        self.learning_rate = 0.001
        self.model = self.build_model(learning_rate=self.learning_rate)

        self.memory_size = 10000
        self.memory = deque(maxlen=self.memory_size)

        self.epsilon = 1
        self.epsilon_decay = 0.995

        self.discount = 0.99

    def build_model(self, learning_rate):
        model = Sequential()
        model.add(layers.InputLayer(input_shape=8))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(4, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
        return model
    
    def act(self, state):
        if self.epsilon > random.random():
            return env.action_space.sample()
        else: 
            return np.argmax(self.model.predict(state))
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, new_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.discount * np.max(self.model.predict(new_state))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1)
        self.epsilon -= self.epsilon_decay

        
agent = DQNAgent()
        
# PARAMATERS
n_episodes = 2000
batch_size = 64

# TRAINING
rewards_history = {}
for i in range(n_episodes):
    state, info = env.reset()
    state = np.reshape(state, [1, 8])
    env.render()
    done = False
    total_reward = 0
    
    while not done:
        env.render()
        action = agent.act(state)
        new_state, reward, done, _, _ = env.step(action)
        new_state = np.reshape(new_state, [1, 8])
        agent.remember(state, action, reward, new_state, done)
        state = new_state
        total_reward += reward
    
    print(f"episode: {i+1}, reward = {total_reward}, epsilon = {int(agent.epsilon)}")
    rewards_history[i+1] = total_reward

    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

# plotting
plt.plot(list(rewards_history.keys()), list(rewards_history.values()))
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Rewards version')
plt.savefig('version.png')  # Save the plot as a PNG image
plt.show()

# Save the agent model and parameters
agent.model.save('agent_model.h5')

# Save the agent's hyperparameters to a JSON file
import json
agent_hyperparams = {
    'learning_rate': agent.learning_rate,
    'discount': agent.discount,
    'epsilon_decay': agent.epsilon_decay,
    'n_episodes': n_episodes,
    'memory_size': agent.memory_size,
    'batch_size': batch_size,
    'epsilon': agent.epsilon,
    'num of hidden layers': 2,
    'hidden layer 1 neurons': 128,
    'hidden layer 2 neurons': 128
}
with open('version.json', 'w') as outfile:
    json.dump(agent_hyperparams, outfile)
