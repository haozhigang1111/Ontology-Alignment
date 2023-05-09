import torch
import torch.nn as nn
import json
import numpy as np
import math
import torch.nn.functional as F
from torchsummary import summary
import scipy.spatial
from functions import *
import argparse


class SiameseMLP(nn.Module):
    def __init__(self,drop_out ,sequence_length1,sequence_length2,channel_num,hidden_size):
        super(SiameseMLP,self).__init__()
        self.length1= sequence_length1
        self.length2 =sequence_length2
        self.channel  = channel_num
        self.FC1 = nn.Linear(sequence_length1*channel_num,hidden_size,bias=True)
        nn.init.xavier_uniform_(self.FC1.weight)


        self.FC2 = nn.Linear(sequence_length2*channel_num,hidden_size,bias=True)
        nn.init.xavier_uniform_(self.FC2.weight)

        self.drop_out1 = nn.Dropout(drop_out)
        self.batchnorm1 = nn.BatchNorm1d(self.length1*hidden_size)
        self.batchnorm2 = nn.BatchNorm1d(self.length2*hidden_size)

    def forward(self,X1,X2):
        x1 = torch.tensor(X1.reshape(-1,self.length1*self.channel))
        #x1 = self.batchnorm1(x1)
        y1 = self.FC1(x1)
        x2 = torch.tensor(X2.reshape(-1,self.length2*self.channel))
        #x2 = self.batchnorm2(x2)
        y2 = self.FC2(x2)

        return y1,y2

class self_att_layer(nn.Module):
    def __init__(self, act_func, emb_size,hid_dim):
        super(self_att_layer,self).__init__()

        self.act_func = act_func
        #print(type(emb_size*2),type(hid_dim))
        self.con1 = nn.Conv1d(emb_size*2,hid_dim,1,bias=False)
        self.con2 = nn.Conv1d(hid_dim,1,1)
        self.con3 = nn.Conv1d(hid_dim,1,1)


    def forward(self,inlayer,adj_mat):
        in_fts = self.con1(torch.unsqueeze(inlayer,0).permute(0,2,1))
        f_1 = torch.reshape(self.con2(in_fts), (-1, 1))
        f_2 = torch.reshape(self.con3(in_fts), (-1, 1))
        logits = f_1 + f_2.T

        adj_tensor = torch.tensor(adj_mat, dtype=torch.float32)
        logits = torch.multiply(adj_tensor, logits)
        bias_mat = -1e9 * (1.0 - (adj_mat > 0))
        coefs = F.softmax(F.leaky_relu(logits) + torch.Tensor(bias_mat),dim=1)

        vals = torch.matmul(coefs, inlayer)
        if self.act_func is None:
            return vals
        else:
            return self.act_func(vals)

class dual_att_layer(nn.Module):
    def __init__(self, act_func, emb_size,hid_dim):
        super(dual_att_layer,self).__init__()

        self.act_func = act_func
        self.con1 = nn.Conv1d(emb_size*2,hid_dim,1,bias=False)
        self.con2 = nn.Conv1d(hid_dim,1,1)
        self.con3 = nn.Conv1d(hid_dim,1,1)


    def forward(self,inlayer,inlayer2,adj_mat):
        in_fts = self.con1(torch.unsqueeze(inlayer2,0).permute(0,2,1))
        f_1 = torch.reshape(self.con2(in_fts), (-1, 1))
        f_2 = torch.reshape(self.con3(in_fts), (-1, 1))
        logits = f_1 + f_2.T

        adj_tensor = torch.tensor(adj_mat, dtype=torch.float32)
        logits = torch.multiply(adj_tensor, logits)
        bias_mat = -1e9 * (1.0 - (adj_mat > 0))
        coefs = F.softmax(F.leaky_relu(logits) + torch.Tensor(bias_mat),dim=1)

        vals = torch.matmul(coefs, inlayer)
        if self.act_func is None:
            return vals
        else:
            return self.act_func(vals)



class sparse_att_layer(nn.Module):
    def __init__(self,emb_size,act_func):
        super(sparse_att_layer,self).__init__()
        self.conv = nn.Conv1d(emb_size*2,1,1)
        self.act_func = act_func

    def forward(self,inlayer,dual_layer,r_mat):
        dual_transform = torch.reshape(self.conv(torch.unsqueeze(dual_layer,0).permute(0,2,1)),(-1,1))
        logits = torch.reshape(embedding_lookup(dual_transform, r_mat.coalesce().values()), [-1])
        lrelu = torch.sparse_coo_tensor(r_mat.coalesce().indices(),
                                F.leaky_relu(logits),
                                (r_mat.shape))

        coefs = torch.sparse.softmax(lrelu,dim=1)
        vals = torch.sparse.mm(coefs, inlayer)
        if self.act_func is None:
            return vals
        else:
            return self.act_func(vals)

class gcn_layer(nn.Module):
    def __init__(self,dimension,dropout,act_fun):
        super(gcn_layer,self).__init__()
        self.act_func = act_fun
        self.W = torch.nn.Parameter(torch.ones([1,dimension],requires_grad=True))
        self.dropout = nn.Dropout(dropout)

    def forward(self,inlayer,M):
        inlayer = self.dropout(inlayer)
        tosum = torch.sparse.mm(M,torch.multiply(inlayer,self.W))
        if self.act_func is None:
            return tosum
        else:
            return self.act_func(tosum)

class highway_layer(nn.Module):
    def __init__(self,dimension):
        super(highway_layer,self).__init__()
        init_range = np.sqrt(6.0 / (dimension + dimension))
        self.kernel_gate = torch.nn.Parameter(torch.tensor(torch.Tensor(dimension,dimension),requires_grad=True))
        self.kernel_gate.data.uniform_(-init_range, init_range)
        self.bias_gate = torch.nn.Parameter(torch.zeros([dimension],requires_grad=True))
    def forward(self,layer1,layer2):
        transform_gate = torch.matmul(layer1, self.kernel_gate) + self.bias_gate
        transform_gate = torch.sigmoid(transform_gate)
        carry_gate = 1.0 - transform_gate
        return transform_gate * layer2 + carry_gate * layer1


class RDGCN(nn.Module):
    def __init__(self,dimension,act_fun,alpha,beta,gamma,k,path,e,KG):
        super(RDGCN,self).__init__()
        self.dimension = dimension
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.primal_X_0 = get_inputlayer(path)
        self.KG = KG
        self.k = k
        self.e =e


        #self.dual_X_1, self.dual_A_1 = get_dual_input(self.primal_X_0, self.head, self.tail, self.head_r, self.tail_r, self.dimension)
        self.self_att_layer = self_att_layer(act_fun, self.dimension,self.dimension//2)
        self.sparse_att_layer1 = sparse_att_layer(self.dimension,act_fun)
        self.dual_att_layer = dual_att_layer(act_fun, self.dimension, self.dimension // 2)
        self.sparse_att_layer2 = sparse_att_layer(self.dimension,act_fun )
        self.gcn1 = gcn_layer(dimension=self.dimension,dropout=0.3,act_fun=act_fun)
        self.gcn2 = gcn_layer(dimension=self.dimension,dropout=0.2,act_fun=act_fun)
        self.highway1 = highway_layer(dimension=self.dimension)
        self.highway2 = highway_layer(dimension=self.dimension)


    def forward(self):
        head, tail, head_r, tail_r, r_mat = rfunc(self.KG, self.e)

        dual_X_1, dual_A_1 = get_dual_input(self.primal_X_0, head, tail, head_r, tail_r,
                                                      self.dimension)
        dual_H_1 = self.self_att_layer(dual_X_1, dual_A_1)
        #print(dual_H_1.shape)
        primal_H_1 = self.sparse_att_layer1(self.primal_X_0, dual_H_1, r_mat)

        primal_X_1 = self.primal_X_0 + self.alpha * primal_H_1

        dual_X_2, dual_A_2 = get_dual_input(primal_X_1, head, tail, head_r, tail_r,
                                                      self.dimension)

        dual_H_2 = self.dual_att_layer(dual_H_1, dual_X_2, dual_A_2)
        primal_H_2 = self.sparse_att_layer2(primal_X_1, dual_H_2, r_mat)
        primal_X_2 = self.primal_X_0 + self.beta * primal_H_2

        M, M_arr = get_sparse_tensor(self.e, self.KG)

        gcn_layer_1 = self.gcn1(primal_X_2,  M )
        gcn_layer_1 = self.highway1(primal_X_2, gcn_layer_1)
        gcn_layer_2 = self.gcn2(gcn_layer_1, M)
        output_layer = self.highway2(gcn_layer_1, gcn_layer_2)

        #loss = get_loss(output_layer, ILL, self.gamma, self.k)
        return output_layer


class Siamese_GCN(nn.Module):
    def __init__(self,drop_out ,sequence_length1,sequence_length2,channel_num,hidden_size,act_fun,alpha,beta,gamma,k,path,e,KG,flag):
        super(Siamese_GCN, self).__init__()
        self.siamese =  SiameseMLP(drop_out ,sequence_length1,sequence_length2,channel_num,hidden_size)
        self.RDGCN = RDGCN(hidden_size,act_fun,alpha,beta,gamma,k,path,e,KG)
        self.highway1  =highway_layer(hidden_size)
        self.highway2 = highway_layer(hidden_size)
        self.liner = nn.Linear(hidden_size*2,hidden_size,bias=False)
        self.f = flag


    def forward(self,X1,X2,x1_id,x2_id):
        #print(X1.shape)
        emb1,emb2 = self.siamese(X1,X2)
        outputlayer = self.RDGCN()
        gcn_x1 = embedding_lookup(outputlayer,x1_id)
        gcn_x2 = embedding_lookup(outputlayer,x2_id)

        ####highway
        if self.f == 'highway':
            emb1 = self.highway1(emb1,gcn_x1)
            emb2 = self.highway2(emb2,gcn_x2)
        #emb1=gcn_x1
        #emb2=gcn_x2


        #####concat
        elif self.f =='concat':
            emb1 = torch.cat((emb1,gcn_x1),dim=-1)
            emb2 = torch.cat((emb2, gcn_x2), dim=-1)
            emb1 = self.liner(emb1)
            emb2 = self.liner(emb2)

        #####add
        else:
            emb1 = emb1+gcn_x1
            emb2 = emb2 + gcn_x2


        return emb1,emb2,outputlayer


