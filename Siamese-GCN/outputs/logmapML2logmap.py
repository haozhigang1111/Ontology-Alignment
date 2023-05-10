import pandas as pd
import argparse


parser = argparse.ArgumentParser()
flag = 'mouse-human'

parser.add_argument('--input_file',type=str,default='siamese_gcn_outputs/'+flag+'_emb_AML.txt')
parser.add_argument('--output',type=str,default='siamese_gcn_outputs/'+flag+'_aml.txt')

FLAGS,unparsed = parser.parse_known_args()

uri1='http://bioontology.org/projects/ontologies/fma/fmaOwlDlComponent_2_0#'
uri2 = 'http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#'

if __name__ == "__main__":


    fw = open(FLAGS.output,'w',encoding='utf8')
    with open(FLAGS.input_file) as f:
        lines = f.readlines()
        for j in range(0, len(lines), 3):
            tmp = lines[j].split('|')

            fw.write(tmp[1].replace('fma:',uri1)+'|'+tmp[2].replace('nci:',uri2)+'|'+tmp[3])
            #fw.write(tmp[1]+'|'+tmp[2]+'|'+tmp[3])
    fw.close()
