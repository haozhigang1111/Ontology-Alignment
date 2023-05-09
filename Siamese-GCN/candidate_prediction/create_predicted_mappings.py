
flag= 'mouse-human'
root = '../data/'+flag+'/'

owllist1 =[]
with open(root+'ent_ids_1','r',encoding='utf8') as f:
    for line in f:
        line = line.strip()
        id,name = line.split('\t')
        owllist1.append(name)

owllist2 =[]
with open(root+'ent_ids_2','r',encoding='utf8') as f:
    for line in f:
        line = line.strip()
        id,name = line.split('\t')
        owllist2.append(name)
fw = open('conference_prediction/'+flag+'.txt','w',encoding='utf8')

for i in range(len(owllist1)):
    for j in range(len(owllist2)):
        fw.write(owllist1[i]+'|'+owllist2[j]+'\n')
