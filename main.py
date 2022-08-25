import numpy as np
from agent import DQNAgent
from envs.static import MEDAEnv
import matplotlib.pyplot as plt
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

env = MEDAEnv()
best_score = -np.inf
load_checkpoint = False
start_time = datetime.now().replace(microsecond=0)

N_GAMES = 1000
N_EPOCHES = 400

agent = DQNAgent(gamma=0.99, epsilon=1.0, lr=0.0001,
					input_dims=env.observation_shape,
					n_actions=4, mem_size=50000, eps_min=0.1,
					batch_size=64, replace=1000, eps_dec=1e-7,
					chkpt_dir='models/', env_name='static_0825')

if load_checkpoint:
	agent.load_models()

fname = agent.env_name + '_' + str(N_EPOCHES) + 'epoches'
figure_file = 'plots/' + fname + '.png'

n_steps = 0
scores, eps_history, steps_array = [], [], []
n_modules = 0
avg_score = 0

for i in range(N_EPOCHES):
	for j in range(N_GAMES):
		done = False
		score = 0
		observation = env.reset()

		while not done:
			action = agent.choose_action(observation)
			observation_, reward, done, _ = env.step(action)

			score += reward
			agent.store_transition(observation, action, reward, observation_, done)
			agent.learn()
			observation = observation_

		scores.append(score)
		steps_array.append(i)

		avg_score = np.mean(scores[-100:])

		if avg_score > best_score:
			if not load_checkpoint:
				agent.save_models()
			best_score = avg_score

		eps_history.append(agent.epsilon)

	if (i % 1 == 0):
		print('epsode ', i, 'average score %.1f best score %.1f epsilon %.2f' % (avg_score, best_score, agent.epsilon),
		 'Elapsed Time  : ', datetime.now().replace(microsecond=0) - start_time)
		


fig, ax1 = plt.subplots()
ax1.plot(avg_score, color = 'red')
ax1.set_xlabel("Epoches")
ax1.set_ylabel("Rewards", color = 'red')

ax2 = ax1.twinx()
ax2.plot(eps_history, color = 'blue')
ax2.set_ylabel("epsilon", color = 'blue')
plt.savefig(figure_file)
