'''
Most of the code is derived from LogMap-ML:


'''



import xml.etree.ElementTree as ET
import argparse
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings('ignore')
#from lib.Label import uri_prefix
"""
Given a file of scored mappings, and an OAEI reference mapping (complete gold standard) file, 
Output Precision, Recall and F1 Score
"""

parser = argparse.ArgumentParser()

flag='mouse-human'
method = 'outputs'
left=flag.split('-')[0]
right=flag.split('-')[1]

parser.add_argument('--prediction_out_file', type=str, default= method+'/'+flag+'_aml.txt')
parser.add_argument('--oaei_GS_file', type=str, default='owl_align/'+flag+'.rdf')
parser.add_argument('--anchor_mapping_file', type=str, default='logmap_outputs/'+flag+'/logmap_anchors.txt')
parser.add_argument('--embedding_file', type=str, default='owl_embedding/'+flag+'_emb')
parser.add_argument('--left_class_file', type=str, default='rdf2rdf/'+left+'_class.list')
parser.add_argument('--right_class_file', type=str, default='rdf2rdf/'+right+'_class.list')
parser.add_argument('--threshold', type=float, default=0.517)
parser.add_argument('--add_anchor', type=bool, default=False)
FLAGS, unparsed = parser.parse_known_args()

owl2vec = Word2Vec.load(FLAGS.embedding_file)

with open(FLAGS.left_class_file) as f:
    left_class = eval(f.readline())

with open(FLAGS.right_class_file) as f:
    right_class = eval(f.readline())

#print(left_class)

def read_oaei_mappings(file_name):
    tree = ET.parse(file_name)
    mappings_str = list()
    all_mappings_str = list()
    for t in tree.getroot().getchildren():
        for m in t.getchildren():
            if 'map' in m.tag:
                for c in m.getchildren():
                    mapping = list()
                    mv = '?'
                    for i, v in enumerate(c.getchildren()):
                        if i < 2:
                            for value in v.attrib.values():
                                mapping.append(value.replace('confof','confOf'))
                                break
                        if i == 3:
                            mv = v.text
                    # if mapping[0] not in left_class or mapping[1] not in right_class:
                    #     continue
                    all_mappings_str.append('|'.join(mapping))
                    if not mv == '?':
                        mappings_str.append('|'.join(mapping))
    return mappings_str, all_mappings_str


if __name__ == "__main__":

    ref_mappings_str, ref_all_mappings_str = read_oaei_mappings(file_name=FLAGS.oaei_GS_file)
    print(len(ref_mappings_str),len(ref_all_mappings_str))

    ref_excluded_mappings_str = set(ref_all_mappings_str) - set(ref_mappings_str)

    anchor_mappings_str = list()
    with open(FLAGS.anchor_mapping_file) as f:
        for line in f.readlines():
            tmp = line.strip().split('|')
            anchor_mappings_str.append('%s|%s' % (tmp[0], tmp[1]))

    pred_mappings_str = list()
    with open(FLAGS.prediction_out_file) as f:
        lines = f.readlines()
        for j in range(0, len(lines)):
            tmp = lines[j].split('|')

            if tmp[0] not in left_class or tmp[1] not in right_class:
                continue
            if len(tmp)==4:
                if float(tmp[3]) >= FLAGS.threshold:
                    pred_mappings_str.append('%s|%s' % (tmp[0], tmp[1]))
            elif len(tmp)==3:
                if float(tmp[2]) >= FLAGS.threshold:
                    pred_mappings_str.append('%s|%s' % (tmp[0], tmp[1]))
            elif len(tmp)==5:
                if float(tmp[3]) >= FLAGS.threshold:
                    pred_mappings_str.append('%s|%s' % (tmp[0], tmp[1]))
            elif len(tmp)==2:

                pred_mappings_str.append('%s|%s' % (tmp[0].replace('\n',''), tmp[1].replace('\n','')))
            else:
                if float(tmp[3]) >= FLAGS.threshold:
                    pred_mappings_str.append('%s|%s' % (tmp[1], tmp[2]))

    #pred_mappings_str,_ = read_oaei_mappings(file_name=FLAGS.prediction_out_file)
    pred_mappings_str = set(pred_mappings_str)

    if FLAGS.add_anchor:
        for a in anchor_mappings_str:
            if a not in pred_mappings_str:
                pred_mappings_str.append(a)

    recall_num = 0
    for s in ref_mappings_str:
        if s in pred_mappings_str:
            recall_num += 1

    print('recall num:',recall_num)
    R = recall_num / len(ref_mappings_str)
    print(len(ref_mappings_str))
    precision_num = 0
    num = 0
    for s in pred_mappings_str:
        if s not in ref_excluded_mappings_str:
            if s in ref_mappings_str:
                precision_num += 1
        num += 1
    print('precise num:',precision_num)
    P = precision_num / num
    print(num)
    F1 = 2 * P * R / (P + R)
    print('Mapping #: %d, Precision: %.3f, Recall: %.3f, F1: %.3f' % (len(pred_mappings_str), P, R, F1))
