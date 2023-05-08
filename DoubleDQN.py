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
        self.online_q_network = self.create_q_network()
        self.target_q_network = self.create_q_network()

        self.memory_size = 10000
        self.memory = deque(maxlen=self.memory_size)

        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        self.discount = 0.99

    def create_q_network(self):
        model = Sequential()
        model.add(layers.InputLayer(input_shape=8))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(4, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model
    
    def act(self, state):
        if self.epsilon > random.random():
            return env.action_space.sample()
        else: 
            return np.argmax(self.online_q_network.predict(state, verbose=0))
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, new_state, done in minibatch:
            target = reward
            if not done:
                best_action = np.argmax(self.online_q_network.predict(new_state, verbose=0))
                target = reward + self.discount * self.target_q_network.predict(new_state, verbose=0)[0][best_action]
            target_f = self.online_q_network.predict(state, verbose=0)
            target_f[0][action] = target
            self.online_q_network.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def copy_weights(self):
        self.target_q_network.set_weights(self.online_q_network.get_weights())
        
agent = DQNAgent()
        
# PARAMATERS
n_episodes = 300
batch_size = 32
max_steps = 1000
update_weights_timestep_count = 0

# TRAINING
rewards_history = {}
for i in range(n_episodes):
    state, info = env.reset()
    state = np.reshape(state, [1, 8])
    env.render()
    done = False
    total_reward = 0
    
    for step in range(max_steps):
        env.render()
        action = agent.act(state)
        new_state, reward, done, _, _ = env.step(action)
        new_state = np.reshape(new_state, [1, 8])
        agent.remember(state, action, reward, new_state, done) 
        state = new_state
        total_reward += reward
        update_weights_timestep_count += 1
        if update_weights_timestep_count % 2000 == 0:
            agent.copy_weights()
        if done:
            break

    print(f"episode: {i+1}, reward = {total_reward}, epsilon = {int(agent.epsilon)}")

    rewards_history[i+1] = total_reward

    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

# plotting
plt.plot(list(rewards_history.keys()), list(rewards_history.values()))
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Rewards Double DQN')
plt.savefig('Double_DQN.png')  
plt.show()

# Save the agent model and parameters
agent.target_q_network.save('Double_DQN(trgt).h5')

# Save the agent's hyperparameters to a JSON file
import json
agent_hyperparams = {
    'method': 'double DQN',
    'learning rate/discount': '0.001/0.99',
    'n_episodes': '300',
    'epsilon decay': '*= 0.995'
}
with open('Double_DQN.json', 'w') as outfile:
    json.dump(agent_hyperparams, outfile)
