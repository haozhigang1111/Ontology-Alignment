'''
The code is derived from LogMap_ML:


'''

import json
from lib.Label import uri_prefix,prefix_uri

with open('data/helis_class_name.json') as  f:
    left_label = json.load(f)

with open('data/foodon_class_name.json') as  f:
    right_label = json.load(f)

fw = open('SGCN_samples_a_lable4.txt','w',encoding='utf8')
with open('SGCN_samples_a_4.txt',encoding='utf8') as f:
    lines = f.readlines()
    for i in range(0,len(lines),3):
        #print(lines[i])
        tmp =lines[i].split('|')
        left_c,right_c,score = tmp[0],tmp[1],tmp[2]
        #print(left_c,right_c)
        left = uri_prefix(left_c)
        right = uri_prefix(right_c)
        #print(left)

        left_l = left_label[left]

        right_l = right_label[right]
        mapping = '%s|%s|%s'%(prefix_uri(left_c),prefix_uri(right_c),str(score))
        fw.write(mapping)
        fw.write(str(left_l)+'|'+str(right_l)+'\n')
        fw.write('\n')

