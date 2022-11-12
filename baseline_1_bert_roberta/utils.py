import numpy as np


def simple_accuracy(preds, labels):
    preds = np.argmax(preds, axis=-1)
    acc = (preds == labels).mean()
    return {"accuracy": round(acc, 4)}


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


def figer_metric(out, l):

    def f1(p, r):
        if r == 0.:
            return 0.
        return 2 * p * r / float(p + r)

    def loose_macro(true, pred):
        num_entities = len(true)
        p = 0.
        r = 0.
        for true_labels, predicted_labels in zip(true, pred):
            if len(predicted_labels) > 0:
                p += len(set(predicted_labels).intersection(set(true_labels))) / float(len(predicted_labels))
            if len(true_labels):
                r += len(set(predicted_labels).intersection(set(true_labels))) / float(len(true_labels))
        precision = p / num_entities
        recall = r / num_entities
        return round(f1(precision, recall), 4)

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
        return round(f1(precision, recall), 4)

    cnt = 0
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
        cnt += set(yy1) == set(yy2)  # 要完全预测对
    acc = round(cnt / l.shape[0], 4)
    macro_F1 = loose_macro(y2, y1)
    micro_F1 = loose_micro(y2, y1)
    rel = {'accuracy': acc, 'macro_F1': macro_F1, 'micro_F1': micro_F1}
    return rel


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
    elif task_name == 'openentity':
        return openentity_metric(preds, labels)
    elif task_name == 'figer':
        return figer_metric(preds, labels)
    elif task_name == 'tacred':
        NO_RELATION = 0 # 0 is No_relation
        label_set = list(range(42))
        label_set.remove(NO_RELATION)
        return tacred_metric(preds, labels, label_set)
    elif task_name == 'fewrel':
        return fewrel_metric(preds, labels)
    else:
        raise KeyError(task_name)
