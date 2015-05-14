from __future__ import division
__author__ = 'Michael May'

#from sklearn.metrics import log_loss
import numpy as np


#labels matrix indictaes if the data is in a matrix or a list
def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    predictions = np.clip(y_pred, eps, 1 - eps)
    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]
    if not y_pred.shape == y_true.shape:
        actual = np.zeros(y_pred.shape)
        rows = actual.shape[0]
        actual[np.arange(rows), y_true.astype(int)] = 1
    else:
        actual = y_true
        rows = actual.shape[0]

    vsota = np.sum(actual * np.log(predictions))
    return -1.0 / rows * vsota


def main():

    pred = [
        [0.05,0.05,0.05,0.8,0.05],
        [0.73,0.05,0.01,0.20,0.02],
        [0.02,0.03,0.01,0.75,0.19],
        [0.01,0.02,0.83,0.12,0.02]
        ]
    classes = [3,0,3,2]
    print multiclass_log_loss(np.array(classes), np.array(pred))


if __name__ == '__main__':
    main()