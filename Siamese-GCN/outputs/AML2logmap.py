import pandas as pd
import argparse


parser = argparse.ArgumentParser()
flag = 'FMA-NCI'
parser.add_argument('--input_file',type=str,default='AML_outputs/'+flag+'.tsv')
parser.add_argument('--output',type=str,default='AML_outputs/'+flag+'.txt')

FLAGS,unparsed = parser.parse_known_args()

if __name__ == "__main__":
    fw = open(FLAGS.output,'w',encoding='utf8')
    with open(FLAGS.input_file,encoding='utf8') as f:
       for line in f:
           parts = line.split('\t')
           if len(parts) ==6 and parts[0] !='Source URI':
                #print(parts)
                fw.write(parts[0]+'|'+parts[2]+'|'+parts[4]+'\n')
    fw.close()
