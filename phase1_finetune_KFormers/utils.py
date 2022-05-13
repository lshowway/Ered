# import logging
# import os
# import sys
# import torch
# import numpy as np
# import argparse
# import re
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import precision_recall_curve, auc, roc_curve, accuracy_score, matthews_corrcoef, f1_score
# from scipy.stats import pearsonr, spearmanr
# from collections import Counter
#
#
# def word_tokenize(sent):
#     pat = re.compile(r'[\w]+|[.,!?;|]')
#     if isinstance(sent, str):
#         return pat.findall(sent.lower())
#     else:
#         return []
#
#
# def str2bool(v):
#     if isinstance(v, bool):
#         return v
#     if v.lower() in ("yes", "true", "t", "y", "1"):
#         return True
#     elif v.lower() in ("no", "false", "f", "n", "0"):
#         return False
#     else:
#         raise argparse.ArgumentTypeError("Boolean value expected.")
#
#
# def init_hvd_cuda(enable_hvd=True, enable_gpu=True):
#     hvd = None
#     if enable_hvd:
#         import horovod.torch as hvd
#
#         hvd.init()
#         logging.info(
#             f"hvd_size:{hvd.size()}, hvd_rank:{hvd.rank()}, hvd_local_rank:{hvd.local_rank()}"
#         )
#
#     hvd_size = hvd.size() if enable_hvd else 1
#     hvd_rank = hvd.rank() if enable_hvd else 0
#     hvd_local_rank = hvd.local_rank() if enable_hvd else 0
#
#     if enable_gpu:
#         torch.cuda.set_device(hvd_local_rank)
#
#     return hvd_size, hvd_rank, hvd_local_rank
#
#
# def setuplogging():
#     root = logging.getLogger()
#     root.setLevel(logging.INFO)
#     handler = logging.StreamHandler(sys.stdout)
#     handler.setLevel(logging.INFO)
#     formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s")
#     handler.setFormatter(formatter)
#     root.addHandler(handler)
#
#
# def dump_args(args):
#     for arg in dir(args):
#         if not arg.startswith("_"):
#             logging.info(f"args[{arg}]={getattr(args, arg)}")
#
#
# def acc(y_true, y_hat):
#     y_hat = torch.argmax(y_hat, dim=-1)
#     tot = y_true.shape[0]
#     hit = torch.sum(y_true == y_hat)
#     return hit.data.float() * 1.0 / tot
#
#
# def dcg_score(y_true, y_score, k=10):
#     order = np.argsort(y_score)[::-1]
#     y_true = np.take(y_true, order[:k])
#     gains = 2 ** y_true - 1
#     discounts = np.log2(np.arange(len(y_true)) + 2)
#     return np.sum(gains / discounts)
#
#
# def ndcg_score(y_true, y_score, k=10):
#     best = dcg_score(y_true, y_true, k)
#     actual = dcg_score(y_true, y_score, k)
#     return actual / best
#
#
# def mrr_score(y_true, y_score):
#     order = np.argsort(y_score)[::-1]
#     y_true = np.take(y_true, order)
#     rr_score = y_true / (np.arange(len(y_true)) + 1)
#     return np.sum(rr_score) / np.sum(y_true)
#
#
# def quality_control_metric(preds, labels, positive_label=1):
#     def _auc(preds, labels):
#         y = np.array(labels)
#         preds = np.array(preds)
#         preds = preds[:, positive_label]  # 1 is positive label
#         fpr, tpr, thresholds = roc_curve(y, preds, pos_label=1)
#         precision, recall, _thresholds = precision_recall_curve(y, preds)
#         roc_auc = auc(fpr, tpr)
#         pr_auc = auc(recall, precision)
#         return {
#             "roc_auc": round(roc_auc, 4),
#             "pr_auc": round(pr_auc, 4)
#         }
#
#     def accuracy(preds, labels):
#         outputs = np.argmax(preds, axis=1)
#         acc = np.sum(outputs == labels) / len(labels)
#         return {"accuracy": round(acc, 4)}
#
#     t1 = _auc(preds, labels)
#     t2 = accuracy(preds, labels)
#     t1.update(t2)
#     return t1
#
#
# # ======================================
# def simple_accuracy(preds, labels):
#     preds = np.argmax(preds, axis=-1)
#     acc = (preds == labels).mean()
#     return {"accuracy": round(acc, 4)}
#
#
# def auc_metrics(preds, labels):
#     y = np.array(labels)
#     pred = np.array(preds)
#     fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1)
#     precision, recall, _thresholds = precision_recall_curve(y, pred)
#     roc_auc = auc(fpr, tpr)
#     pr_auc = auc(recall, precision)
#     return {
#         "roc_auc": roc_auc,
#         "pr_auc": pr_auc
#     }
#
#
# def acc_and_f1(preds, labels):
#     acc = simple_accuracy(preds, labels)
#     preds = np.argmax(preds, axis=-1)
#     f1 = f1_score(y_true=labels, y_pred=preds)
#     acc.update({
#         "f1": f1,
#         "acc_and_f1": (acc['acc'] + f1) / 2,
#     })
#     return acc
#
#
# def pearson_and_spearman(preds, labels):
#     pearson_corr = pearsonr(preds, labels)[0]
#     spearman_corr = spearmanr(preds, labels)[0]
#     return {
#         "pearson": pearson_corr,
#         "spearmanr": spearman_corr,
#         "corr": (pearson_corr + spearman_corr) / 2,
#     }



def openentity_metric(out, l):

    def f1(p, r):
        if r == 0.:
            return 0.
        return 2 * p * r / float(p + r)


    def loose_micro(true, pred):
        num_predicted_labels = 0.
        num_true_labels = 0.
        num_correct_labels = 0.
        for true_labels, predicted_labels in zip(true, pred):
            num_predicted_labels += len(predicted_labels)
            num_true_labels += len(true_labels)
            num_correct_labels += len(set(predicted_labels).intersection(set(true_labels)))
        if num_predicted_labels > 0:
            precision = num_correct_labels / num_predicted_labels
        else:
            precision = 0.
        recall = num_correct_labels / num_true_labels
        return round(precision, 4), round(recall, 4), round(f1(precision, recall), 4)

    y1 = []
    y2 = []
    for x1, x2 in zip(out, l):
        yy1 = []
        yy2 = []
        for i in range(len(x1)):
            if x1[i] > 0:
                yy1.append(i)
            if x2[i] > 0:
                yy2.append(i)
        y1.append(yy1)
        y2.append(yy2)
    p, r, f_value = loose_micro(y2, y1)
    rel = {'precision': p, 'recall': r, 'micro_F1': f_value}
    return rel


# def figer_metric(out, l):
#
#     def f1(p, r):
#         if r == 0.:
#             return 0.
#         return 2 * p * r / float(p + r)
#
#     def loose_macro(true, pred):
#         num_entities = len(true)
#         p = 0.
#         r = 0.
#         for true_labels, predicted_labels in zip(true, pred):
#             if len(predicted_labels) > 0:
#                 p += len(set(predicted_labels).intersection(set(true_labels))) / float(len(predicted_labels))
#             if len(true_labels):
#                 r += len(set(predicted_labels).intersection(set(true_labels))) / float(len(true_labels))
#         precision = p / num_entities
#         recall = r / num_entities
#         return round(f1(precision, recall), 4)
#
#     def loose_micro(true, pred):
#         num_predicted_labels = 0.
#         num_true_labels = 0.
#         num_correct_labels = 0.
#
#         for true_labels, predicted_labels in zip(true, pred):
#             num_predicted_labels += len(predicted_labels)
#             num_true_labels += len(true_labels)
#             num_correct_labels += len(set(predicted_labels).intersection(set(true_labels)))
#         if num_predicted_labels > 0:
#             precision = num_correct_labels / num_predicted_labels
#         else:
#             precision = 0.
#         recall = num_correct_labels / num_true_labels
#         return round(f1(precision, recall), 4)
#
#     cnt = 0
#     y1 = []
#     y2 = []
#     for x1, x2 in zip(out, l):
#         yy1 = []
#         yy2 = []
#         for i in range(len(x1)):
#             if x1[i] > 0:
#                 yy1.append(i)
#             if x2[i] > 0:
#                 yy2.append(i)
#         y1.append(yy1)
#         y2.append(yy2)
#         cnt += set(yy1) == set(yy2)  # 要完全预测对
#     acc = round(cnt / l.shape[0], 4)
#     macro_F1 = loose_macro(y2, y1)
#     micro_F1 = loose_micro(y2, y1)
#     rel = {'accuracy': acc, 'macro_F1': macro_F1, 'micro_F1': micro_F1}
#     return rel
#
#
# def relation_classification_metric(pred_result, labels, na_id=-1):
#     correct = 0
#     total = len(labels)
#     correct_positive = 0
#     pred_positive = 0
#     gold_positive = 0
#
#     for i in range(total):
#         if labels[i] == pred_result[i]:
#             correct += 1
#             if labels[i] != na_id:
#                 correct_positive += 1
#         if labels[i] != na_id:
#             gold_positive += 1
#         if pred_result[i] != na_id:
#             pred_positive += 1
#     acc = float(correct) / float(total)
#     try:
#         micro_p = float(correct_positive) / float(pred_positive)
#     except:
#         micro_p = 0
#     try:
#         micro_r = float(correct_positive) / float(gold_positive)
#     except:
#         micro_r = 0
#     try:
#         micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
#     except:
#         micro_f1 = 0
#     result = {'accuracy': round(acc, 4), 'micro_p': round(micro_p),
#               'micro_r': round(micro_r), 'micro_F1': round(micro_f1)}
#
#     return result
#
#
def fewrel_metric(pred_result, labels):
    num_predicted_labels = 0
    num_gold_labels = 0
    num_correct_labels = 0

    for label, prediction in zip(labels, pred_result):
        num_predicted_labels += 1
        num_gold_labels += 1
        if prediction == label:
            num_correct_labels += 1

    if num_predicted_labels > 0:
        precision = num_correct_labels / num_predicted_labels
    else:
        precision = 0.0
    recall = num_correct_labels / num_gold_labels
    if recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    result = {'micro_p': precision, 'micro_r': recall, 'micro_F1': f1}

    return result


# def tacred_metric(pred_result, labels, na_id):
#     correct = 0
#     total = len(labels)
#     correct_positive = 0
#     pred_positive = 0
#     gold_positive = 0
#
#     for i in range(total):
#         if labels[i] == pred_result[i]:
#             correct += 1
#             if labels[i] != na_id:
#                 correct_positive += 1
#         if labels[i] != na_id:
#             gold_positive += 1
#         if pred_result[i] != na_id:
#             pred_positive += 1
#     acc = float(correct) / float(total)
#     try:
#         micro_p = float(correct_positive) / float(pred_positive)
#     except:
#         micro_p = 0
#     try:
#         micro_r = float(correct_positive) / float(gold_positive)
#     except:
#         micro_r = 0
#     try:
#         micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
#     except:
#         micro_f1 = 0
#
#     result = {'accuracy': round(acc, 4), 'micro_p': round(micro_p),
#               'micro_r': round(micro_r), 'micro_F1': round(micro_f1)}
#     return result

def tacred_metric(predictions, labels, label_set):
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    precision, recall, f1, supp = precision_recall_fscore_support(labels, predictions, pos_label=None, average='micro', labels=label_set)
    acc = accuracy_score(labels, predictions)
    result = {'accuracy': round(acc, 4), 'micro_p': round(precision, 4),
                  'micro_r': round(recall, 4), 'micro_F1': round(f1, 4)}
    return result


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "sst2":
        t1 = simple_accuracy(preds, labels)
        return t1
    elif task_name == 'eem':
        return quality_control_metric(preds, labels)
    elif task_name == 'openentity':
        return openentity_metric(preds, labels)
    elif task_name == 'figer':
        return figer_metric(preds, labels)
    elif task_name == 'tacred':
        from data_utils import TACRED_relations
        NO_RELATION = 0 # 0 is No_relation
        label_set = list(range(42))
        label_set.remove(NO_RELATION)
        return tacred_metric(preds, labels, label_set)
    elif task_name == 'fewrel':
        return fewrel_metric(preds, labels)
    else:
        raise KeyError(task_name)
#
#
# # ======================================
# def load_matrix(embedding_file_path, word_dict, word_embedding_dim):
#     embedding_matrix = np.random.uniform(size=(len(word_dict) + 1,
#                                                word_embedding_dim))
#     have_word = []
#     if embedding_file_path is not None:
#         with open(embedding_file_path, 'rb') as f:
#             while True:
#                 line = f.readline()
#                 if len(line) == 0:
#                     break
#                 line = line.split()
#                 word = line[0].decode()
#                 if word in word_dict:
#                     index = word_dict[word]
#                     tp = [float(x) for x in line[1:]]
#                     embedding_matrix[index] = np.array(tp)
#                     have_word.append(word)
#     return embedding_matrix, have_word
#
#
# def latest_checkpoint(directory):
#     if not os.path.exists(directory):
#         return None
#     all_checkpoints = {
#         int(x.split('.')[-2].split('-')[-1]): x
#         for x in os.listdir(directory)
#     }
#     if not all_checkpoints:
#         return None
#     return os.path.join(directory,
#                         all_checkpoints[max(all_checkpoints.keys())])
#
#
# def get_checkpoint(directory, ckpt_name):
#     ckpt_path = os.path.join(directory, ckpt_name)
#     if os.path.exists(ckpt_path):
#         return ckpt_path
#     else:
#         return None
