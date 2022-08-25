import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
	def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
		super(DeepQNetwork, self).__init__()
		self.checkpoint_dir = chkpt_dir
		self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

		# nn.Conv2(channel of input, filter number, carnel size, stride)
#		print(input_dims[0])
		self.conv1 = nn.Conv2d(input_dims[0], 32, 3, stride=1)
		self.conv2 = nn.Conv2d(32, 64, 2, stride=1)
		self.conv3 = nn.Conv2d(64, 64, 2, stride=1)

		fc_input_dims = self.compute_conv_output_dims(input_dims)

		self.fc1 = nn.Linear(fc_input_dims, 512)
		self.fc2 = nn.Linear(512, n_actions)

		self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
		self.loss = nn.MSELoss()
		self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
#		self.device = T.device('cpu')
		self.to(self.device)

	def compute_conv_output_dims(self, input_dims):
		state = T.zeros(1, *input_dims)
		dims = self.conv1(state)
		dims = self.conv2(dims)
		dims = self.conv3(dims)
		return int(np.prod(dims.size()))

	def forward(self, state):
#		print(state)

#		print("--- State ---")
#		print(state)

		conv1 = F.relu(self.conv1(state))
		conv2 = F.relu(self.conv2(conv1))
		conv3 = F.relu(self.conv3(conv2))
		# conv3's chape is BS * n_filters * H * W
		conv_state = conv3.view(conv3.size()[0], -1)
#		print(conv3.shape)
		flat1 = F.relu(self.fc1(conv_state))
		actions = self.fc2(flat1)

		return actions

	def save_checkpoint(self):
		print('...saveing checkpoint ...')
		#print(self.checkpoint_file)
		T.save(self.state_dict(), self.checkpoint_file)

	def load_checkpoint(self):
		print('...loading checkpoint ...')
		self.load_state_dict(T.load(self.checkpoint_file))
