import numpy as np
from agent import DQNAgent
from envs.static import MEDAEnv as Static
from envs.dynamic import MEDAEnv as Dynamic

import warnings
warnings.filterwarnings("ignore")

def _gen_random_map():
	map = np.random.choice([".", "#", '*'], (8, 8), p=[0.8, 0.1, 0.1])
	map[0][0] = "D"
	map[-1][-1] = "G"

	print("--- Start map ---")
	print(map)

	return map

map = _gen_random_map()

# Static
env = Static(test_flag=True)
agent = DQNAgent(gamma=0.99, epsilon=0, lr=0.0001, input_dims=env.observation_shape,
				 n_actions=4, mem_size=50000, eps_min=0, batch_size=64, replace=1000, eps_dec=0,
				 chkpt_dir='models/', env_name='test')
agent.load_models()

done = False
score = 0
observation = env.reset(test_map=map)

print('---------Static Result -----------')
while not done:
	action = agent.choose_action(observation)
	observation_, reward, done, _ = env.step(action)
	score += reward
	observation = observation_

# Dynamic
env = Dynamic(test_flag=True)
agent = DQNAgent(gamma=0.99, epsilon=0, lr=0.0001, input_dims=env.observation_shape,
				 n_actions=4, mem_size=50000, eps_min=0, batch_size=64, replace=1000, eps_dec=0,
				 chkpt_dir='models/', env_name='test')
agent.load_models()

done = False
score = 0
observation = env.reset(test_map=map)

print('---------Dynamic Result -----------')
while not done:
	action = agent.choose_action(observation)
	observation_, reward, done, _ = env.step(action)
	score += reward
	observation = observation_
