import torch
import torch.nn as nn
import json
import numpy as np
import math
import torch.nn.functional as F
from torchsummary import summary
import scipy.spatial
from functions import *
from model import Siamese_GCN
import argparse



parser = argparse.ArgumentParser()

flag = 'mouse-human'
model_class='mouse-human_emb'
emb_flag = 'mouse-human'
# l_class = flag.split('-')[0]
# r_class = flag.split('-')[1]


parser.add_argument('--train_path_file', type=str, default='data/'+flag+'/mappings_train.txt')
parser.add_argument('--valid_path_file', type=str, default='data/'+flag+'/mappings_valid.txt')
parser.add_argument('--class_word_size', type=int, default=14)
parser.add_argument('--left_path_size', type=int, default=7)
parser.add_argument('--right_path_size', type=int, default=31)
parser.add_argument('--left_w2v_dir', type=str, default='owl_embedding/'+emb_flag+'_emb')
parser.add_argument('--right_w2v_dir', type=str, default='owl_embedding/'+emb_flag+'_emb')
parser.add_argument('--embedding_type', type=str, default='owl2vec')
parser.add_argument('--vec_type', type=str, default='word-uri')
parser.add_argument('--path_type', type=str, default='uri+label')
parser.add_argument('--encoder_type', type=str, default='class-con')
parser.add_argument('--mlp_hidden_size', type=int, default=768)
parser.add_argument('--num_epochs', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--nn_dir', type=str, default='checkpoints/'+model_class+'/')
FLAGS, unparsed = parser.parse_known_args()



if __name__ =="__main__":

    from include.Config import Config
    from include.Load import *
    from gensim.models import Word2Vec
    import datetime


    start = datetime.datetime.now()

    e1 = len(set(loadfile(Config.e1, 1)))  # 实体数量
    e2 = len(set(loadfile(Config.e2, 1)))
    e = e1 +e2
 

    KG1 = loadfile(Config.kg1, 3)  # 图谱1中的三元组
    KG2 = loadfile(Config.kg2, 3)


    if FLAGS.embedding_type == 'owl2vec':
        left_wv_model = Word2Vec.load(FLAGS.left_w2v_dir)
        right_wv_model = Word2Vec.load(FLAGS.right_w2v_dir)
    else:
        left_wv_model = json.load(open(FLAGS.left_w2v_dir))
        right_wv_model = json.load(open(FLAGS.right_w2v_dir))

    train_X1, train_X2, train_Y, train_num, X1_id,X2_id = load_samples(file_name=FLAGS.train_path_file, FLAGS=FLAGS,
                                                          left_wv_model=left_wv_model,
                                                          right_wv_model=right_wv_model,
                                                            left_owl_id='data/'+flag+'/ent_ids_1',
                                                            right_owl_id='data/'+flag+'/ent_ids_2')

    shuffle_indices = np.random.permutation(np.arange(train_num))
    train_X1, train_X2, train_Y,X1_id,X2_id = train_X1[shuffle_indices], train_X2[shuffle_indices], train_Y[shuffle_indices],X1_id[shuffle_indices],X2_id[shuffle_indices]
    valid_X1, valid_X2, valid_Y, valid_num,valid_X1_id,valid_X2_id = load_samples(file_name=FLAGS.valid_path_file, FLAGS=FLAGS,
                                                          left_wv_model=left_wv_model,
                                                          right_wv_model=right_wv_model,
                                                              left_owl_id='data/'+flag+'/ent_ids_1',
                                                              right_owl_id='data/'+flag+'/ent_ids_2'
                                                              )


    model = Siamese_GCN(0.2 ,3,3,768,768,F.relu,Config.alpha, Config.beta, Config.gamma, Config.k,'data/'+flag+'/embeddings',e,KG1 + KG2,flag='highway')

    indices = np.where(train_Y[:, 1] > 0)
    ILL = list(zip(X1_id[indices[0]], X2_id[indices[0]]))

    siamese_nn_train(model, train_X1, train_X2, train_Y,X1_id,X2_id, FLAGS,ILL,Config.k,Config.gamma,index=e1,data=[valid_X1, valid_X2,  valid_X1_id,valid_X2_id,valid_Y])

    siamese_nn_valid(valid_X1, valid_X2,  valid_X1_id,valid_X2_id,valid_Y,os.path.join(FLAGS.nn_dir,'model'))

    end = datetime.datetime.now()
    print((end - start).seconds/60, 'minutes')
