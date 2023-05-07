import gym
from keras.models import Sequential
from keras import layers 
from keras.optimizers import Adam
from collections import deque
import random
import numpy as np

# ENVIRONMENT
env = gym.make("LunarLander-v2")

# AGENT
class DQNAgent:
    def __init__(self):
        self.model = self.build_model(learning_rate=0.001)

        self.memory = deque(maxlen=1000)

        self.epsilon = 1
        self.epsilon_decay = 0.001

        self.discount = 0.95

    def build_model(self, learning_rate):
        model = Sequential()
        model.add(layers.InputLayer(input_shape=8))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(4, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
        return model
    
    def act(self, state):
        if self.epsilon > random.random():
            return env.action_space.sample()
        else: 
            return np.argmax(self.model.predict(state))
    
    def remember(self, state, action, reward, new_state):
        self.memory.append(state, action, reward, new_state)

    def replay(self):
        minibatch = random.sample(self.memory, 32)
        for state, action, reward, new_state in minibatch:
            target = reward
            if not done:
                target = reward + self.discount * np.max(self.model.predict(new_state))
            target_f = self.model.predict(state)
            target_f[action][0] = target
            self.model.fit(state, target_f, epochs=1)
        self.epsilon -= self.epsilon_decay

        
agent = DQNAgent()
        
# PARAMATERS
n_episodes = 500
batch_size = 32

# TESTING
for i in range(n_episodes):
    state, info = env.reset()
    env.render()
    done = False
    total_reward = 0
    
    while not done:
        env.render()
        action = agent.act(state)
        new_state, reward, done, _, _ = env.step(action)
        agent.remember(state, action, reward, new_state)
        state = new_state
        total_reward += reward
    
    print(f"episode: {i+1}, reward = {total_reward}, epsilon = {int(agent.epsilon)}")

    if len(agent.memory) > batch_size:
        agent.replay()