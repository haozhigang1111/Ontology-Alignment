import os
import argparse
import json
from gensim.models import Word2Vec
from functions import *
parser = argparse.ArgumentParser()
import torch
from model import self_att_layer,dual_att_layer,sparse_att_layer,gcn_layer,highway_layer,Siamese_GCN,SiameseMLP,RDGCN

flag = 'mouse-human'
emb_flag = 'mouse-human'
l_class = flag.split('-')[0]
r_class = flag.split('-')[1]
model_class='mouse-human_emb'



parser.add_argument('--left_path_file', type=str, default='class&path/'+l_class+'_all_paths.txt')
parser.add_argument('--right_path_file', type=str, default='class&path/'+r_class+'_all_paths.txt')
parser.add_argument('--left_class_name_file', type=str, default='class&path/'+l_class+'_class_name.json')
parser.add_argument('--right_class_name_file', type=str, default='class&path/'+r_class+'_class_name.json')
parser.add_argument('--candidate_file', type=str, default='candidate_prediction/'+'mouse-human.txt')
parser.add_argument('--prediction_out_file', type=str, default='outputs/'+model_class+'_AML.txt')
parser.add_argument('--class_word_size', type=int, default=6)
parser.add_argument('--left_path_size', type=int, default=3)
parser.add_argument('--right_path_size', type=int, default=3)
parser.add_argument('--left_w2v_dir', type=str, default='embedding/'+emb_flag+'_emb')
parser.add_argument('--right_w2v_dir', type=str, default='embedding/'+emb_flag+'_emb')
parser.add_argument('--embedding_type', type=str, default='owl2vec')
parser.add_argument('--path_type', type=str, default='uri+label',
                    help='three settings: label, path, uri+label;'
                         'label: the class embedding as input; '
                         'path: the path embedding as input'
                         'uri+label: the uri name and label of the class')

parser.add_argument('--vec_type', type=str, default='word-uri')
parser.add_argument('--keep_uri', type=str, default='yes')

parser.add_argument('--encoder_type', type=str, default='class-con')
parser.add_argument('--nn_dir', type=str, default='checkpoints/'+model_class)
parser.add_argument('--nn_type', type=str, default=model_class)
FLAGS, unparsed = parser.parse_known_args()

FLAGS.nn_dir = os.path.join(FLAGS.nn_dir, 'model')

left_paths = [line.strip().split(',') for line in open(FLAGS.left_path_file).readlines()]
right_paths = [line.strip().split(',') for line in open(FLAGS.right_path_file).readlines()]
left_names = json.load(open(FLAGS.left_class_name_file,encoding='utf8'))
right_names = json.load(open(FLAGS.right_class_name_file,encoding='utf8'))

mappings, mappings_n = list(), list()
with open(FLAGS.candidate_file,encoding='utf8') as f:
    for i, line in enumerate(f.readlines()):
        #m = line.strip().split(', ')[1] if ', ' in line else line.strip()
        m = line.strip()
        m_split = m.split('|')
        c1 = uri_prefix(uri=m_split[0])
        c2 = uri_prefix(uri=m_split[1])
        n1 = get_label(cls=c1, paths=left_paths, names=left_names, label_type='path',
                       keep_uri=(FLAGS.keep_uri == 'yes'))
        n2 = get_label(cls=c2, paths=right_paths, names=right_names, label_type='path',
                       keep_uri=(FLAGS.keep_uri == 'yes'))

        origin = 'i=%d|%s|%s' % (i + 1, c1, c2)
        name = '%s|%s' % (n1, n2)
        mappings.append(origin)
        mappings_n.append(name)

if FLAGS.embedding_type =='owl2vec':
    left_wv_model = Word2Vec.load(FLAGS.left_w2v_dir)
    right_wv_model = Word2Vec.load(FLAGS.right_w2v_dir)
else:
    left_wv_model = json.load(open(FLAGS.left_w2v_dir))
    right_wv_model = json.load(open(FLAGS.right_w2v_dir))

X1, X2,X1_id,X2_id,mappings = to_samples(mappings=mappings, mappings_n=mappings_n, FLAGS=FLAGS, left_wv_model=left_wv_model,
                    right_wv_model=right_wv_model,left_owl_id='data/'+'mouse-human'+'/ent_ids_1',
                                                            right_owl_id='data/'+'mouse-human'+'/ent_ids_2')
model = torch.load(FLAGS.nn_dir)

test_distances = siamese_nn_predict(X1,X2,X1_id,X2_id,model)
test_scores = 1 - test_distances

with open(FLAGS.prediction_out_file, 'w',encoding='utf8') as f:
    for i, mapping in enumerate(mappings):
        f.write('%s|%.3f\n' % (mapping, test_scores[i]))
        f.write('%s\n' % mappings_n[i])
        f.write('\n')
print('%d mappings, all predicted' % len(mappings))