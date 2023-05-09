import pandas as pd
import argparse


parser = argparse.ArgumentParser()
flag = 'cmt-Conference_2'

parser.add_argument('--input_file',type=str,default='logmap_ML_outputs/'+flag+'.txt')
parser.add_argument('--output',type=str,default='logmap_ML_outputs/'+flag+'_ml.txt')

FLAGS,unparsed = parser.parse_known_args()

#uri1='http://www.fbk.eu/ontologies/virtualcoach#'
#uri2 = 'http://purl.obolibrary.org/obo/'

if __name__ == "__main__":

    FLAGS.output='siamese_gcn_outputs/cmt-sigkdd.txt'
    FLAGS.input_file= 'siamese_gcn_outputs/cmt-sigkdd_ce.txt'

    fw = open(FLAGS.output,'w',encoding='utf8')
    with open(FLAGS.input_file) as f:
        lines = f.readlines()
        for j in range(0, len(lines), 3):
            tmp = lines[j].split('|')

            #fw.write(tmp[1].replace('vc:',uri1)+'|'+tmp[2].replace('obo:',uri2)+'|'+tmp[3])
            fw.write(tmp[1]+'|'+tmp[2]+'|'+tmp[3])
    fw.close()
