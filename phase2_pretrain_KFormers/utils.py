import numpy as np


def acc(y_true, y_hat):
    y_hat = torch.argmax(y_hat, dim=-1)
    tot = y_true.shape[0]
    hit = torch.sum(y_true == y_hat)
    return hit.data.float() * 1.0 / tot



def simple_accuracy(preds, labels):
    preds = np.argmax(preds, axis=-1)
    acc = (preds == labels).mean()
    return {"accuracy": round(acc, 4)}



def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    t1 = simple_accuracy(preds, labels)

    return t1



