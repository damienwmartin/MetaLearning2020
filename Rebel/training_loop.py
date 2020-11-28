from rebel_alg import ReBeL
from originalrebelcode.models import build_mlp
import torch
import torch.functional.nn as F






v_net = build_mlp(input_dim, n_hidden, n_layers, out_dim)




def train(v_net, epochs, games_per_epoch, batch_size=64, game, v_net = None):
	#TODO
	input_dim, output_dim = game.get_dims()
	n_hidden = 128
	n_layers = 4
	opt = torch.optim.Adam(v_net.parameters())
	v_net.train()
	for i in range(epochs):
		train_x = []
		train_y = []
		for j in range(games_per_epoch):
			#Play a full game with rebel
			init_PBS = game.get_init_pbs()
			new_train_data = ReBeL(init_PBS, game, v_net)
			new_train_x, new_train_y = list(zip(*new_train_data))
			train_x.append(new_train_x)
			train_y.append(new_train_y)

		train_x = torch.tensor(train_x)
		train_y = torch.tensor(train_y)
		opt.zero_grad()
		output = v_net.forward(train_x)
		loss = F.mse_loss(output, train_y)
		loss.backward()
		opt.step()

		if i % 10 == 0:
			print(f'Epoch {i}: Loss {loss}')
	







