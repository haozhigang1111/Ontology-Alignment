import torch
import torch.nn as nn


class Config:
	task = 'data/mouse-human' 
	e1 =  task + '/ent_ids_1'
	e2 =  task + '/ent_ids_2'
	kg1 = task + '/triples_1'
	kg2 = task + '/triples_2'
	dim = 768
	act_func = nn.ReLU
	alpha = 0.03
	beta = 0.05
	gamma = 1.0  # margin based loss
	k = 50  # number of negative samples for each positive one
	output = 'gcn_outputs/'+task
	epochs = 30

