import csv
import json
import math
import os
import re

import numpy as np
import scipy.spatial
import torch
import torch.nn.functional as F
from nltk.tokenize import word_tokenize

namespaces = ["http://www.fbk.eu/ontologies/virtualcoach#",
              "http://purl.obolibrary.org/obo/",
              "http://bioontology.org/projects/ontologies/fma/fmaOwlDlComponent_2_0#",
              "http://www.ihtsdo.org/snomed#",
              "http://www.orpha.net/ORDO/",
              "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#",
              'http://www.semanticweb.org/dell/ontologies/2022/4/gb2760-ontology#',
              'http://www.semanticweb.org/dell/ontologies/2022/4/gb2762-ontology#',
              'http://www.semanticweb.org/dell/ontologies/2022/4/gb2763-ontology#']
prefixes = ["vc:", "obo:", "fma:", "snomed:", "ordo:", "nci:", "GB2760:", "GB2762:", "GB2763:"]


def uri_prefix(uri):
    for i, namespace in enumerate(namespaces):
        if namespace in uri:
            return uri.replace(namespace, prefixes[i])
    return uri


def label_preprocess(label):
    if label is None:
        return ''
    else:
        return label.lower().replace('"', '')


def entity_to_string(ent, names):
    name = names[ent]
    if name[1] is None:
        uri_name = name[0]
        name_str = uri_name_to_string(uri_name=uri_name)
    else:
        label = name[1]
        name_str = label_preprocess(label=label)
    return '"%s"' % name_str


def path_to_string(path, names, keep_uri):
    names_ = list()
    for e in path:
        if keep_uri:
            names_.append('"%s"' % e)
        else:
            names_.append(entity_to_string(ent=e, names=names))
    return ','.join(names_)


def get_label(cls, paths, names, label_type, keep_uri=False):
    if label_type == 'path':
        for p in paths:
            if cls in p:
                path = p[p.index(cls):]
                return path_to_string(path=path, names=names, keep_uri=keep_uri)
        names[cls] = [cls.split('#')[-1], None]
        # print(cls,names)
        return path_to_string(path=[cls], names=names, keep_uri=keep_uri)
    else:
        return path_to_string(path=[cls], names=names, keep_uri=keep_uri)
    return '""'


def prefix_uri(ns_uri):
    for i, prefix in enumerate(prefixes):
        if prefix in ns_uri:
            return ns_uri.replace(prefix, namespaces[i])
    return ns_uri


def uri_name_to_string(uri_name):
    """parse the URI name (camel cases)"""
    uri_name = uri_name.replace('_', ' ').replace('-', ' ').replace('.', ' '). \
        replace('/', ' ').replace('"', ' ').replace("'", ' ')
    words = []
    for item in uri_name.split():
        matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', item)
        for m in matches:
            word = m.group(0)
            words.append(word.lower())
    return ' '.join(words)


def get_inputlayer(path='embedding/helis-foodon.json'):
    with open(path, mode='r', encoding='utf-8') as f:
        embedding_list = json.load(f)
        print(len(embedding_list), 'rows,', len(embedding_list[0]), 'columns.')

    input_embedings = torch.nn.Parameter(torch.tensor(embedding_list, requires_grad=True))
    #input_embedings = torch.nn.Parameter(torch.randn(len(embedding_list),len(embedding_list[0])),requires_grad=True)
    return input_embedings


def get_mat(e, KG):
    du = [{e_id} for e_id in range(e)]  # du记录相关联实体的集合
    for tri in KG:
        if tri[0] != tri[2]:
            du[tri[0]].add(tri[2])  # 与tri[0]关联的实体tri[2]
            du[tri[2]].add(tri[0])
    du = [len(d) for d in du]  # 记录集合的大小
    M = {}  # 记录两个实体间是否有关系
    for tri in KG:
        if tri[0] == tri[2]:
            continue
        if (tri[0], tri[2]) not in M:
            M[(tri[0], tri[2])] = 1
        else:
            pass
        if (tri[2], tri[0]) not in M:
            M[(tri[2], tri[0])] = 1
        else:
            pass

    for i in range(e):
        M[(i, i)] = 1
    return M, du


def get_sparse_tensor(e, KG):
    # print('getting a sparse tensor...')
    M, du = get_mat(e, KG)  # M记录两个实体间是否有关系,du记录每个实体相关联的实体集合大小
    ind = []
    val = []
    M_arr = np.zeros((e, e))
    for fir, sec in M:
        ind.append((sec, fir))
        val.append(M[(fir, sec)] / math.sqrt(du[fir]) / math.sqrt(du[sec]))
        M_arr[fir][sec] = 1.0
    M = torch.sparse_coo_tensor(torch.LongTensor(ind).T, torch.FloatTensor(val), torch.Size([e, e]))  # 记录两个实体间的关联强度

    return M, M_arr


def rfunc(KG, e):
    head = {}
    tail = {}
    cnt = {}
    for tri in KG:
        if tri[1] not in cnt:  # 第一次记录关系tri[1]
            cnt[tri[1]] = 1
            head[tri[1]] = set([tri[0]])
            tail[tri[1]] = set([tri[2]])
        else:
            cnt[tri[1]] += 1  # 关系tri[1]已有记录,相应次数加1
            head[tri[1]].add(tri[0])  # 记录关系tri[1]的头实体
            tail[tri[1]].add(tri[2])  # 记录关系tri[1]的尾实体
    r_num = len(head)  # 关系种类
    head_r = np.zeros((e, r_num))
    tail_r = np.zeros((e, r_num))
    r_mat_ind = []
    r_mat_val = []
    for tri in KG:  # 595
        # print(tri)
        head_r[tri[0]][tri[1]] = 1  # 头实体与关系
        tail_r[tri[2]][tri[1]] = 1  # 尾实体与关系
        r_mat_ind.append([tri[0], tri[2]])
        r_mat_val.append(tri[1])
    # print(len(r_mat_val),len(r_mat_ind),e)
    r_mat = torch.sparse_coo_tensor(list(zip(*r_mat_ind)), r_mat_val, (e, e))  # 记录实体间的关系种类

    return head, tail, head_r, tail_r, r_mat


def compute_r(inlayer, head_r, tail_r, dimension):
    head_l = torch.tensor(head_r, dtype=torch.float32).T
    tail_l = torch.tensor(tail_r, dtype=torch.float32).T
    # print(head_l.shape) #（e，r）->(r,e)
    # print(tail_r.shape)
    L = torch.matmul(head_l, inlayer) / \
        torch.unsqueeze(torch.sum(head_l, axis=-1), -1)  # 将所有头实体向量相加然后除以头实体总数，Ci的计算方式
    R = torch.matmul(tail_l, inlayer) / \
        torch.unsqueeze(torch.sum(tail_l, axis=-1), -1)
    r_embeddings = torch.cat([L, R], axis=-1)
    return r_embeddings


def get_dual_input(inlayer, head, tail, head_r, tail_r, dimension):
    # print(head_r.shape)
    dual_X = compute_r(inlayer, head_r, tail_r, dimension)  # 初始关系嵌入
    # print('computing the dual input...')
    count_r = len(head)
    dual_A = np.zeros((count_r, count_r))
    for i in range(count_r):
        for j in range(count_r):
            a_h = len(head[i] & head[j]) / len(head[i] | head[j])
            a_t = len(tail[i] & tail[j]) / len(tail[i] | tail[j])
            dual_A[i][j] = a_h + a_t
    return dual_X, dual_A


def embedding_lookup(tensor, ids):
    size = tensor.shape[0]
    index = np.array(ids)
    # print(ids)
    embedding = tensor[index]
    return embedding


def get_loss(outlayer, ILL, gamma, k, neg_left, neg_right, neg2_left, neg2_right):
    #print('getting loss...')
    left = ILL[:, 0]
    right = ILL[:, 1]
    t = len(ILL)
    left_x = embedding_lookup(outlayer, left)
    right_x = embedding_lookup(outlayer, right)

    A = torch.sum(torch.abs(left_x - right_x), 1)
    # A = tf.reduce_sum(get_cosine(left_x,right_x))

    neg_l_x = embedding_lookup(outlayer, neg_left)
    neg_r_x = embedding_lookup(outlayer, neg_right)

    B = torch.sum(torch.abs(neg_l_x - neg_r_x), 1)
    C = - torch.reshape(B, [t, k])
    D = A + gamma
    L1 = F.relu(torch.add(C, torch.reshape(D, [t, 1])))

    neg_l_x = embedding_lookup(outlayer, neg2_left)
    neg_r_x = embedding_lookup(outlayer, neg2_right)

    B = torch.sum(torch.abs(neg_l_x - neg_r_x), 1)
    C = - torch.reshape(B, [t, k])
    L2 = F.relu(torch.add(C, torch.reshape(D, [t, 1])))

    return (torch.sum(L1) + torch.sum(L2)) / (2.0 * k * t)


def get_neg(ILL, output_layer, k, index, flag):
    neg = []
    t = len(ILL)
    ILL_vec = np.array([output_layer[e1].detach().numpy() for e1 in ILL])
    KG_vec = np.array(output_layer.detach().numpy())
    sim = scipy.spatial.distance.cdist(ILL_vec, KG_vec, metric='euclidean')
    for i in range(t):
        neg_list = []
        rank = sim[i, :].argsort()
        if flag == 'l':
            for j in rank:
                if j < index:
                    neg_list.append(j)
        else:
            for j in rank:
                if j >= index:
                    neg_list.append(j)
        neg.append(neg_list[0:k])

    neg = np.array(neg)
    neg = neg.reshape((t * k,))
    return neg


def batch_iter(data, batch_size, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches = int((data_size - 1) / batch_size) + 1
    if shuffle:
        batch_shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[batch_shuffle_indices]
    else:
        shuffled_data = data

    if num_batches > 0:
        for batch_num in range(num_batches):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
    else:
        yield shuffled_data


def siamese_nn_train(model, train_x1, train_x2, y_train, x1_id, x2_id, FLAGS, ILL, k, gamma, index,data):
    t = len(ILL)
    ILL = np.array(ILL)
    L = np.ones((t, k)) * (ILL[:, 0].reshape((t, 1)))
    neg_left = L.reshape((t * k,))
    L = np.ones((t, k)) * (ILL[:, 1].reshape((t, 1)))
    neg2_right = L.reshape((t * k,))

    y_train = y_train[:, 1]
    X1, X2, X1_id, X2_id,Y = data[0], data[1], data[2], data[3],data[4]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    flag = False
    for epoch in range(FLAGS.num_epochs):
        model.train()
        batches = batch_iter(list(zip(train_x1, train_x2, y_train, x1_id, x2_id)), FLAGS.batch_size)
        if epoch %5 ==0:
           flag =True

        for i, batch in enumerate(batches):
            if i != 0:
                model.RDGCN.eval()
            x1_batch, x2_batch, y_batch, id1_batch, id2_batch = zip(*batch)
            out1, out2, out = model(torch.tensor(x1_batch, dtype=torch.float32),
                                    torch.tensor(x2_batch, dtype=torch.float32), id1_batch, id2_batch)

            # CE
            CE_loss, dis = get_loss2(out1, out2, torch.tensor(y_batch, dtype=torch.float32))

            #distance
            if flag:
                neg2_left = get_neg(ILL[:, 1], out, k, flag='l', index=index)
                neg_right = get_neg(ILL[:, 0], out, k, flag='r', index=index)
                flag=False

            dis_loss = get_loss(out, ILL, gamma, k, neg_left, neg_right, neg2_left, neg2_right)
            loss = 0.7*CE_loss + 0.3*dis_loss
            #loss = CE_loss

            acc = get_Acc(dis, torch.tensor(y_batch, dtype=torch.float32))

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            print('batch loss:', loss.detach().numpy(), 'batch Acc:', acc.detach().numpy())

        # model.eval()
        # out1, out2, _ = model(torch.tensor(train_x1, dtype=torch.float32), torch.tensor(train_x2, dtype=torch.float32),
        #                       x1_id, x2_id)
        # loss, dis = get_loss2(out1, out2, torch.tensor(y_train, dtype=torch.float32))
        # acc = get_Acc(dis, torch.tensor(y_train, dtype=torch.float32))
        #print('epoch %d:' % (epoch), 'loss:', loss, 'Acc:', acc)
        print('epoch %d:' % (epoch))
        #
        #
        # distance = siamese_nn_predict(X1, X2, X1_id, X2_id, model)
        # valid_scores = 1 - distance
        # threshold, f1, p, r, acc = threshold_searching(Y=Y[:, 1], scores=valid_scores, num=len(distance))
        # print('\n ##### best setting, threshold: %.2f, precision: %.3f, recall: %.3f, f1: %.3f, acc: %.3f ##### \n' % (
        #     threshold, p, r, f1, acc))

    if not os.path.exists(FLAGS.nn_dir):
        os.makedirs(FLAGS.nn_dir)
    model_path = os.path.join(FLAGS.nn_dir, 'model')
    torch.save(model, model_path)


def siamese_nn_predict(X1, X2, X1_id, X2_id, model):
    model.eval()
    x1_emb, x2_emb, _ = model(torch.tensor(X1, dtype=torch.float32), torch.tensor(X2, dtype=torch.float32), X1_id,
                              X2_id)
    distance = torch.sqrt(torch.sum(torch.square(torch.subtract(x1_emb, x2_emb)), 1, keepdim=True))
    denominator = torch.add(torch.sqrt(torch.sum(torch.square(x1_emb), 1, keepdim=True)),
                            torch.sqrt(torch.sum(torch.square(x2_emb), 1, keepdim=True)))
    distance = torch.div(distance, denominator)
    distance = torch.reshape(distance, [-1])
    return distance


def cal_result_score(Y, scores, alpha, n):
    pos, true_pos, true_neg, false_neg = 0, 0, 0, 0
    for i in range(n):
        score = scores[i]
        y = Y[i]
        if score >= alpha:
            pos += 1
            if y == 1:
                true_pos += 1
        if score < alpha and y == 0:
            true_neg += 1
        if score < alpha and y == 1:
            false_neg += 1
    precision = true_pos / pos if pos > 0 else 0
    recall = true_pos / (true_pos + false_neg) if true_pos + false_neg > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    acc = (true_pos + true_neg) / n
    return precision, recall, f1, acc


def threshold_searching(Y, scores, num):
    max_valid_f1, max_valid_p, max_valid_r, max_valid_acc, max_alpha = -np.inf, -np.inf, -np.inf, -np.inf, 0
    for alpha in np.arange(0, 1, 0.01):
        valid_p, valid_r, valid_f1, valid_acc = cal_result_score(Y=Y, scores=scores, alpha=alpha, n=num)
        print('alpha: %.2f, precision: %.4f, recall: %.4f, f1 score: %.4f, accuracy: %.4f'
              % (alpha, valid_p, valid_r, valid_f1, valid_acc))
        if valid_f1 > max_valid_f1:
            max_alpha, max_valid_f1, max_valid_p, max_valid_r, max_valid_acc = alpha, valid_f1, valid_p, valid_r, valid_acc
    return max_alpha, max_valid_f1, max_valid_p, max_valid_r, max_valid_acc


def siamese_nn_valid(X1, X2, X1_id, X2_id, Y, model_dir):
    model = torch.load(model_dir)
    distance = siamese_nn_predict(X1, X2, X1_id, X2_id, model)
    valid_scores = 1 - distance
    threshold, f1, p, r, acc = threshold_searching(Y=Y[:, 1], scores=valid_scores, num=len(distance))
    print('\n ##### best setting, threshold: %.2f, precision: %.4f, recall: %.4f, f1: %.4f, acc: %.4f ##### \n' % (
        threshold, p, r, f1, acc))


def get_hits(vec, test_pair, top_k=(1, 10, 50, 100)):
    Lvec = np.array([vec[e1] for e1, e2 in test_pair])
    Rvec = np.array([vec[e2] for e1, e2 in test_pair])
    sim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cosine')
    sim = np.diagonal(sim)
    score = 1 - sim
    print('avg_score:', np.array(score).mean())


def gcn_training(model, learning_rate, epochs, ILL, train_Y, X1_id, X2_id, e, k, gamma, index, model_path, loss_type):
    t = len(ILL)
    ILL = np.array(ILL)
    L = np.ones((t, k)) * (ILL[:, 0].reshape((t, 1)))
    neg_left = L.reshape((t * k,))
    L = np.ones((t, k)) * (ILL[:, 1].reshape((t, 1)))
    neg2_right = L.reshape((t * k,))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for i in range(epochs):
        model.train()
        out = model()

        if i % 10 == 0:
            neg2_left = get_neg(ILL[:, 1], out, k, flag='l', index=index)
            neg_right = get_neg(ILL[:, 0], out, k, flag='r', index=index)

        out1 = embedding_lookup(out, X1_id)
        out2 = embedding_lookup(out, X2_id)
        y_train = train_Y[:, 1]
        if loss_type == 'distance':
            loss = get_loss(out, ILL, gamma, k, neg_left, neg_right, neg2_left, neg2_right)
            _, dis = get_loss2(out1, out2, torch.tensor(y_train, dtype=torch.float32))
        elif loss_type == 'CE':
            loss, dis = get_loss2(out1, out2, torch.tensor(y_train, dtype=torch.float32))
        else:
            loss1 = get_loss(out, ILL, gamma, k, neg_left, neg_right, neg2_left, neg2_right)
            loss2, dis = get_loss2(out1, out2, torch.tensor(y_train, dtype=torch.float32))
            loss = 0.7 * loss2 + 0.3 * loss1

        acc = get_Acc(dis, torch.tensor(y_train, dtype=torch.float32))
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        '''
        if i%2 ==0:
            model.eval()
            out = model()
            get_hits(out.detach().numpy(), test)
        '''
        print('loss: ', loss, 'Acc:', acc)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_path = os.path.join(model_path, 'model')
    torch.save(model, model_path)

    return out


def gcn_predict(X1_id, X2_id, model):
    model.eval()
    output = model()
    x1_emb = embedding_lookup(output, X1_id)
    x2_emb = embedding_lookup(output, X2_id)
    distance = torch.sqrt(torch.sum(torch.square(torch.subtract(x1_emb, x2_emb)), 1, keepdim=True))
    denominator = torch.add(torch.sqrt(torch.sum(torch.square(x1_emb), 1, keepdim=True)),
                            torch.sqrt(torch.sum(torch.square(x2_emb), 1, keepdim=True)))
    distance = torch.div(distance, denominator)
    distance = torch.reshape(distance, [-1])
    return distance


def gcn_valid(X1_id, X2_id, Y, model_dir):
    model = torch.load(model_dir)
    distance = gcn_predict(X1_id, X2_id, model)
    valid_scores = 1 - distance
    threshold, f1, p, r, acc = threshold_searching(Y=Y[:, 1], scores=valid_scores, num=len(distance))
    print('\n ##### best setting, threshold: %.2f, precision: %.3f, recall: %.3f, f1: %.3f, acc: %.3f ##### \n' % (
        threshold, p, r, f1, acc))


def get_loss2(emb1, emb2, input_y):
    #print(emb1,emb2)
    distance = torch.sqrt(torch.sum(torch.square(torch.subtract(emb1, emb2)), 1, keepdim=True))
    denominator = torch.add(torch.sqrt(torch.sum(torch.square(emb1), 1, keepdim=True)),
                            torch.sqrt(torch.sum(torch.square(emb2), 1, keepdim=True)))
    distance = torch.div(distance, denominator)
    distance = torch.reshape(distance, [-1])
    #print(distance)
    item1 = input_y * torch.square(distance)
    item2 = (1 - input_y) * torch.square(torch.maximum((1 - distance), torch.tensor(0)))
    loss = torch.sum(item1 + item2) / 2 / len(input_y)

    return loss, distance


def get_Acc(distance, input_y):
    temp_sim = torch.subtract(torch.ones_like(distance), (distance > 0.5).float())
    #print(temp_sim)
    correct_predictions = (temp_sim == input_y)
    accuracy = torch.mean(correct_predictions.float())
    return accuracy


def to_words(item):
    if item.startswith('http://'):
        if '#' in item:
            uri_name = item.split('#')[1]
        else:
            uri_name = item.split('/')[-1]
        words_str = uri_name_to_string(uri_name=uri_name)
        words = words_str.split(' ')
    else:
        item = item.replace('_', ' ').replace('-', ' ').replace('.', ' ').replace('/', ' '). \
            replace('"', ' ').replace("'", ' ').replace('\\', ' ').replace('(', ' ').replace(')', ' ')
        tokenized_line = ' '.join(word_tokenize(item))
        # words = [word for word in tokenized_line.lower().split() if word.isalpha()]
        words = [word for word in tokenized_line.lower().split()]
    return words


def URI_Vector(cls, wv_model, embedding_type):
    cls_uri = prefix_uri(ns_uri=cls)
    if embedding_type == 'word2vec':
        if cls_uri in wv_model.wv:
            return wv_model.wv[cls_uri]
        else:
            return np.zeros(wv_model.vector_size)
    else:
        if cls_uri in wv_model:
            return wv_model[cls_uri]
        else:
            return np.zeros(768)


def PathEncoderWordAvg(name_path, wv_model, embedding_type):
    if embedding_type=='owl2vec':
        wv_dim = wv_model.vector_size
    else:
        wv_dim = 768
    num, v = 0, np.zeros(wv_dim)
    for item in name_path:
        for word in to_words(item=item):
            if embedding_type == 'owl2vec':
                if word in wv_model.wv:
                    num += 1
                    v += wv_model.wv[word]
            else:
                if word in wv_model:
                    num += 1
                    v += wv_model[word]
    avg = (v / num) if num > 0 else v
    return avg


def PathEncoderWordCon(name_path, class_num, word_num, wv_model, embedding_type):
    # name_path.reverse()
    wv_dim = wv_model.vector_size
    name_path = name_path[0:class_num] if len(name_path) >= class_num else name_path + ['NaN'] * (
            class_num - len(name_path))

    sequence = list()
    for item in name_path:
        words = to_words(item=item)
        words = words[0:word_num] if len(words) >= word_num else words + ['NaN'] * (word_num - len(words))
        sequence = sequence + words

    e = np.zeros((len(sequence), wv_dim))
    for i, word in enumerate(sequence):
        if embedding_type == 'owl2vec':
            if word == 'NaN' or word not in wv_model.wv:
                e[i, :] = np.zeros(wv_dim)
            else:
                e[i, :] = wv_model.wv[word]
        else:
            if word == 'NaN' or word not in wv_model:
                e[i, :] = np.zeros(wv_dim)
            else:
                e[i, :] = wv_model[word]
    return e


def PathEncoderClassCon(name_path, class_num, wv_model, embedding_type):
    # name_path.reverse()
    if embedding_type == 'owl2vec':
        wv_dim = wv_model.vector_size
    else:
        wv_dim = 768
    name_path = name_path[0:class_num] if len(name_path) >= class_num else name_path + ['NaN'] * (
            class_num - len(name_path))

    e = np.zeros((len(name_path), wv_dim))
    for i, item in enumerate(name_path):
        if item == 'NaN':
            e[i, :] = np.zeros(wv_dim)
        else:
            e[i, :] = PathEncoderWordAvg(name_path=[item], wv_model=wv_model,embedding_type=embedding_type)

    return e


def PathEncoderAvg(cls, name_path, wv_model, vec_type, embedding_type):
    if vec_type == 'word':
        return PathEncoderWordAvg(name_path=name_path, wv_model=wv_model, embedding_type=embedding_type)
    elif vec_type == 'uri':
        return URI_Vector(cls=cls, wv_model=wv_model)
    else:
        word_avg = PathEncoderWordAvg(name_path=name_path, wv_model=wv_model, embedding_type=embedding_type)
        uri_avg = URI_Vector(cls=cls, wv_model=wv_model)
        return np.concatenate((word_avg, uri_avg))


def load_samples(file_name, FLAGS, left_wv_model, right_wv_model, left_owl_id, right_owl_id):
    x1_dict = {}
    with open(left_owl_id, encoding='utf8') as f:
        for line in f:
            line = line.strip()
            id, owl = line.split('\t')
            x1_dict[owl] = int(id)

    x2_dict = {}
    with open(right_owl_id, encoding='utf8') as f:
        for line in f:
            line = line.strip()
            id, owl = line.split('\t')
            x2_dict[owl] = int(id)

    if FLAGS.path_type == 'label':
        FLAGS.left_path_size = 1
        FLAGS.right_path_size = 1
    if FLAGS.path_type == 'uri+label':
        FLAGS.left_path_size = 3
        FLAGS.right_path_size = 3
    if FLAGS.embedding_type == 'owl2vec':
        left_wv_dim = left_wv_model.vector_size
        right_wv_dim = right_wv_model.vector_size
    else:
        left_wv_dim = 768
        right_wv_dim = 768
    lines = open(file_name).readlines()
    num = int(len(lines) / 3)
    if FLAGS.encoder_type == 'word-con':
        X1 = np.zeros((num, FLAGS.left_path_size * FLAGS.class_word_size, left_wv_dim))
        X2 = np.zeros((num, FLAGS.right_path_size * FLAGS.class_word_size, right_wv_dim))
    elif FLAGS.encoder_type == 'class-con':
        X1 = np.zeros((num, FLAGS.left_path_size, left_wv_dim))
        X2 = np.zeros((num, FLAGS.right_path_size, right_wv_dim))
    else:  # "avg"
        if FLAGS.vec_type == 'word-uri':
            X1 = np.zeros((num, 1, left_wv_dim * 2))
            X2 = np.zeros((num, 1, right_wv_dim * 2))
        else:
            X1 = np.zeros((num, 1, left_wv_dim))
            X2 = np.zeros((num, 1, right_wv_dim))
    Y = np.zeros((num, 2))

    x1_id = []
    x2_id = []
    for i in range(0, len(lines), 3):
        name_mapping = lines[i + 1]
        tmp = name_mapping.split('|')
        p1 = [x for x in list(csv.reader([tmp[2]], delimiter=',', quotechar='"'))[0]]
        p2 = [x for x in list(csv.reader([tmp[3]], delimiter=',', quotechar='"'))[0]]
        mapping = lines[i].strip().split('|')
        left_c, right_c = mapping[2], mapping[3]
        x1_id.append(x1_dict[prefix_uri(left_c)])
        x2_id.append(x2_dict[prefix_uri(right_c)])

        if FLAGS.path_type == 'label':
            p1 = [p1[0]]
            p2 = [p2[0]]
        if FLAGS.path_type == 'uri+label':
            p1 = [left_c.split(':')[1]] + p1
            p2 = [right_c.split(':')[1]] + p2

        j = int(i / 3)
        if FLAGS.encoder_type == 'word-con':
            X1[j] = PathEncoderWordCon(name_path=p1, wv_model=left_wv_model, class_num=FLAGS.left_path_size,
                                       word_num=FLAGS.class_word_size, embedding_type=FLAGS.embedding_type)
            X2[j] = PathEncoderWordCon(name_path=p2, wv_model=right_wv_model, class_num=FLAGS.right_path_size,
                                       word_num=FLAGS.class_word_size, embedding_type=FLAGS.embedding_type)
        elif FLAGS.encoder_type == 'class-con':
            X1[j] = PathEncoderClassCon(name_path=p1, wv_model=left_wv_model, class_num=FLAGS.left_path_size,
                                        embedding_type=FLAGS.embedding_type)
            X2[j] = PathEncoderClassCon(name_path=p2, wv_model=right_wv_model, class_num=FLAGS.right_path_size,
                                        embedding_type=FLAGS.embedding_type)
        else:  # 'avg'
            X1[j, 0] = PathEncoderAvg(cls=left_c, name_path=p1, wv_model=left_wv_model, vec_type=FLAGS.vec_type,
                                      embedding_type=FLAGS.embedding_type)
            X2[j, 0] = PathEncoderAvg(cls=right_c, name_path=p2, wv_model=right_wv_model, vec_type=FLAGS.vec_type,
                                      embedding_type=FLAGS.embedding_type)
        Y[j] = np.array([1.0, 0.0]) if tmp[0].startswith('neg') else np.array([0.0, 1.0])

    return X1, X2, Y, num, np.array(x1_id), np.array(x2_id)


def to_samples(mappings, mappings_n, FLAGS, left_wv_model, right_wv_model, left_owl_id, right_owl_id):
    x1_dict = {}
    maps=[]
    with open(left_owl_id, encoding='utf8') as f:
        for line in f:
            line = line.strip()
            id, owl = line.split('\t')
            x1_dict[owl] = int(id)

    x2_dict = {}
    with open(right_owl_id, encoding='utf8') as f:
        for line in f:
            line = line.strip()
            id, owl = line.split('\t')
            x2_dict[owl] = int(id)

    if FLAGS.path_type == 'label':
        FLAGS.left_path_size = 1
        FLAGS.right_path_size = 1
    if FLAGS.path_type == 'uri+label':
        FLAGS.left_path_size = 3
        FLAGS.right_path_size = 3

    if FLAGS.embedding_type=='owl2vec':

        left_wv_dim = left_wv_model.vector_size
        right_wv_dim = right_wv_model.vector_size
    else:
        left_wv_dim = 768
        right_wv_dim = 768

    num = len(mappings_n)

    if FLAGS.encoder_type == 'word-con':
        X1 = np.zeros((num, FLAGS.left_path_size * FLAGS.class_word_size, left_wv_dim))
        X2 = np.zeros((num, FLAGS.right_path_size * FLAGS.class_word_size, right_wv_dim))
    elif FLAGS.encoder_type == 'class-con':
        X1 = np.zeros((num, FLAGS.left_path_size, left_wv_dim))
        X2 = np.zeros((num, FLAGS.right_path_size, right_wv_dim))
    else:
        if FLAGS.vec_type == 'word-uri':
            X1 = np.zeros((num, 1, left_wv_dim * 2))
            X2 = np.zeros((num, 1, right_wv_dim * 2))
        else:
            X1 = np.zeros((num, 1, left_wv_dim))
            X2 = np.zeros((num, 1, right_wv_dim))

    x1_id = []
    x2_id = []

    for i in range(num):
        tmp = mappings_n[i].split('|')
        p1 = [x for x in list(csv.reader([tmp[0]], delimiter=',', quotechar='"'))[0]]
        p2 = [x for x in list(csv.reader([tmp[1]], delimiter=',', quotechar='"'))[0]]

        tmp = mappings[i].split('|')
        left_c, right_c = tmp[1], tmp[2]
        # if prefix_uri(left_c) not in x1_dict or prefix_uri(right_c) not in x2_dict:
        #     continue
        maps.append(mappings[i])
        x1 = x1_dict[prefix_uri(left_c)]

        x2 = x2_dict[prefix_uri(right_c)]

        x1_id.append(x1)
        x2_id.append(x2)

        if FLAGS.path_type == 'label':
            p1 = p1[0:1]
            p2 = p2[0:1]
        if FLAGS.path_type == 'uri+label':
            p1 = [left_c.split(':')[1]] + p1
            p2 = [right_c.split(':')[1]] + p2

        if FLAGS.encoder_type == 'word-con':
            X1[i] = PathEncoderWordCon(name_path=p1, wv_model=left_wv_model, class_num=FLAGS.left_path_size,
                                       word_num=FLAGS.class_word_size,embedding_type=FLAGS.embedding_type)
            X2[i] = PathEncoderWordCon(name_path=p2, wv_model=right_wv_model, class_num=FLAGS.right_path_size,
                                       word_num=FLAGS.class_word_size,embedding_type=FLAGS.embedding_type)
        elif FLAGS.encoder_type == 'class-con':
            X1[i] = PathEncoderClassCon(name_path=p1, wv_model=left_wv_model, class_num=FLAGS.left_path_size,embedding_type=FLAGS.embedding_type)
            X2[i] = PathEncoderClassCon(name_path=p2, wv_model=right_wv_model, class_num=FLAGS.right_path_size,embedding_type=FLAGS.embedding_type)
        else:
            X1[i, 0] = PathEncoderAvg(cls=left_c, name_path=p1, wv_model=left_wv_model, vec_type=FLAGS.vec_type,embedding_type=FLAGS.embedding_type)
            X2[i, 0] = PathEncoderAvg(cls=right_c, name_path=p2, wv_model=right_wv_model, vec_type=FLAGS.vec_type,embedding_type=FLAGS.embedding_type)

    X1 = X1[:len(x1_id)]
    X2 = X2[:len(x2_id)]
    return X1, X2, np.array(x1_id), np.array(x2_id),maps
