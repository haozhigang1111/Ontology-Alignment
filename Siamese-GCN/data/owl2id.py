import argparse
from owlready2 import *
import os



parser = argparse.ArgumentParser()
parser.add_argument('--left_onto_file',type=str,default='../owl_data/largebio/FMA.owl')
parser.add_argument('--right_onto_file',type=str,default='../owl_data/largebio/NCI.owl')
parser.add_argument('--output',type=str,default='FMA-NCI')

FLAGS,unparsed = parser.parse_known_args()




def class2id(o1,o2):
    c1_name = dict()
    for id,c in enumerate(o1.classes()):
        c1_name[c.iri] =id
    offset = id+1

    c2_name = dict()
    for id,c in enumerate(o2.classes()):
        c2_name[c.iri] =id+offset
    return c1_name,c2_name

def owl2triples(o,o_dict):
    triples = []
    for c in o.classes():
        print(c)
        subclasses =  o.get_parents_of(c)
        if subclasses:
            #print(subclasses)
            for l_class in subclasses:
                try:
                    l_name = l_class.iri
                    r_name = c.iri
                    if l_name in o_dict and r_name in o_dict:
                        #print(o_dict[l_name],o_dict[r_name])
                        triples.append((o_dict[l_name],o_dict[r_name]))
                except Exception :
                    #print(l_class,'------')
                    pass
    return triples

if __name__ =="__main__":

    output_path = FLAGS.output
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)


    ontofile1 = get_ontology(FLAGS.left_onto_file).load()
    ontofile2 = get_ontology(FLAGS.right_onto_file).load()

    name1,name2 = class2id(ontofile1,ontofile2)
    #print(name1)
    #print(name2)

    l_triples = owl2triples(ontofile1,name1)
    r_triples = owl2triples(ontofile2,name2)

    ref_file_path = FLAGS.reference_file

    ref_list =[]
    with open(ref_file_path,'r',encoding='utf8') as f:
        for line in f:
            line = line.strip()
            parts = line.split('|')
            l_class = parts[0]
            r_class = parts[1]
            ref_list.append((name1[l_class],name2[r_class]))
    print(ref_list)

    with open(output_path+'/ent_ids_1','w',encoding='utf8') as f:
        for k,v in name1.items():
            f.write(str(v)+'\t'+k)
            f.write('\n')


    with open(output_path+'/ent_ids_2','w',encoding='utf8') as f:
        for k,v in name2.items():
            f.write(str(v)+'\t'+k)
            f.write('\n')

    with open(output_path+'/triples_1','w',encoding='utf8') as f:
        for (l,r) in l_triples:
            f.write(str(l)+'\t'+str(0)+'\t'+str(r))
            f.write('\n')

    with open(output_path+'/triples_2','w',encoding='utf8') as f:
        for (l, r) in r_triples:
            f.write(str(l) + '\t' +str(0)+'\t'+ str(r))
            f.write('\n')



