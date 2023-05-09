import argparse
#from lib.Label import uri_prefix
"""
Given a file of scored mappings, and an OAEI reference mapping (complete gold standard) file, 
Output Precision, Recall and F1 Score
"""

parser = argparse.ArgumentParser()
parser.add_argument('--prediction_out_file', type=str, default='AML_outputs/GB2760-GB2762.txt')
parser.add_argument('--oaei_GS_file', type=str, default='logmap_outputs/1_2/GS_file')
parser.add_argument('--threshold', type=float, default=0.5)
FLAGS, unparsed = parser.parse_known_args()


def read_GS_mappings(file_name):

    mappings_str = list()
    with open(file_name,'r',encoding='utf8') as f:
        for line in f:
            line = line.strip()
            mappings_str.append(line)

    return mappings_str


if __name__ == "__main__":

    ref_mappings_str = read_GS_mappings(file_name=FLAGS.oaei_GS_file)


    anchor_mappings_str = list()


    pred_mappings_str = list()
    with open(FLAGS.prediction_out_file,encoding='utf8') as f:
        lines = f.readlines()
        for j in range(0, len(lines)):
            tmp = lines[j].split('|')
            print(tmp)
            if len(tmp)==4 or len(tmp)==5 :
                if float(tmp[3]) >= FLAGS.threshold:
                    pred_mappings_str.append('%s|%s' % (tmp[0], tmp[1]))
            elif len(tmp)==3:
                if float(tmp[2]) >= FLAGS.threshold:
                    pred_mappings_str.append('%s|%s' % (tmp[0], tmp[1]))
            else:
                if float(tmp[3]) >= FLAGS.threshold:
                    pred_mappings_str.append('%s|%s' % (tmp[1], tmp[2]))



    recall_num = 0
    for s in ref_mappings_str:
        if s in pred_mappings_str:
            recall_num += 1
    R = recall_num / len(ref_mappings_str)
    precision_num = 0
    num = 0
    for s in pred_mappings_str:
        if s in ref_mappings_str:
            precision_num += 1
        num += 1
    P = precision_num / num
    F1 = 2 * P * R / (P + R)
    print('Mapping #: %d, Precision: %.3f, Recall: %.3f, F1: %.3f' % (len(pred_mappings_str), P, R, F1))
