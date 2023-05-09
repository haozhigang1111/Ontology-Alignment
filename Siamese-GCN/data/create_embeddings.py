import argparse
import os
from gensim.models import Word2Vec
import json
import re
from bert_serving.client import BertClient
import numpy as np



parser = argparse.ArgumentParser()
flag = 'helis-foodon'

parser.add_argument('--owl2vec_file',type=str,default='../owl_embedding/helis-foodon_emb')
parser.add_argument('--left_owl2id',type=str,default=flag+'/ent_ids_1')
parser.add_argument('--right_owl2id',type=str,default=flag+'/ent_ids_2')
parser.add_argument('--output',type=str,default=flag+'/embeddings_bert')
parser.add_argument('--emb_type',type=str,default='bert')

FLAGS,unparsed = parser.parse_known_args()


def uri_name_to_string(uri_name):
    """parse the URI name (camel cases)"""
    uri_name = uri_name.replace('_', ' ').replace('-', ' ').replace('.', ' ').\
        replace('/', ' ').replace('"', ' ').replace("'", ' ')
    words = []
    for item in uri_name.split():
        matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', item)
        for m in matches:
            word = m.group(0)
            words.append(word.lower())
    return ' '.join(words)


bc = BertClient()

if __name__ =="__main__":
    f_owl2vec = FLAGS.owl2vec_file
    owl2vec = Word2Vec.load(f_owl2vec)
    l_owl2id = FLAGS.left_owl2id
    r_owl2id = FLAGS.right_owl2id
    embeddings=[]
    emb_dict={}

    #print(owl2vec.wv.key_to_index)

    with open(l_owl2id,'r',encoding='utf8') as f:
        for line in f:
            line = line.strip()
            _,Class = line.split()

            #bert
            if FLAGS.emb_type == 'bert':
                c = re.split('[#|/]', Class)[-1]
                new_c = uri_name_to_string(c)
                words = new_c.split()
                food_emb = bc.encode(words)
                emb_dict[Class] = food_emb.mean(axis=0).tolist()
                embeddings.append(emb_dict[Class])



            # #owl2vec
            elif FLAGS.emb_type == 'owl':
                if Class in owl2vec.wv:
                    embeddings.append(owl2vec.wv[Class].tolist())
                    emb_dict[Class] = owl2vec.wv[Class].tolist()
                else:

                    c = re.split('[#|/]',Class)[-1]
                    new_c = uri_name_to_string(c)
                    words = new_c.split()
                    food_emb = bc.encode(words)
                    emb_dict[Class] = food_emb.mean(axis=0).tolist()
                    embeddings.append(emb_dict[Class])

            else:
                c = re.split('[#|/]', Class)[-1]
                new_c = uri_name_to_string(c)
                words = new_c.split()
                l  = len(words)
                emb = np.random.normal(l, 1, size=(1, 768))
                emb_dict[Class] = emb.mean(axis=0).tolist()
                embeddings.append(emb_dict[Class])



    with open(r_owl2id, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            _, Class = line.split()

            # bert
            if FLAGS.emb_type == 'bert':
                c = re.split('[#|/]', Class)[-1]
                new_c = uri_name_to_string(c)
                words = new_c.split()
                food_emb = bc.encode(words)
                emb_dict[Class] = food_emb.mean(axis=0).tolist()
                embeddings.append(emb_dict[Class])



            # #owl2vec
            elif FLAGS.emb_type == 'owl':
                if Class in owl2vec.wv:
                    embeddings.append(owl2vec.wv[Class].tolist())
                    emb_dict[Class] = owl2vec.wv[Class].tolist()
                else:

                    c = re.split('[#|/]', Class)[-1]
                    new_c = uri_name_to_string(c)
                    words = new_c.split()
                    food_emb = bc.encode(words)
                    emb_dict[Class] = food_emb.mean(axis=0).tolist()
                    embeddings.append(emb_dict[Class])

            else:
                c = re.split('[#|/]', Class)[-1]
                new_c = uri_name_to_string(c)
                words = new_c.split()
                l = len(words)
                emb = np.random.normal(l, 1, size=(1, 768))
                emb_dict[Class] = emb.mean(axis=0).tolist()
                embeddings.append(emb_dict[Class])


    print(len(embeddings))
    with open(FLAGS.output, 'w') as f:
        json.dump(embeddings, f)

    print(len(emb_dict))

    if FLAGS.emb_type == 'bert':
        outfile = flag+'_bert.json'
        with open(outfile, 'w') as f:
            json.dump(emb_dict, f)

    elif FLAGS.emb_type == 'random':
        outfile = flag+'_random.json'
        with open(outfile, 'w') as f:
            json.dump(emb_dict, f)
