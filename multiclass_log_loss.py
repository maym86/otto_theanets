from __future__ import division
__author__ = 'Michael May'

import scipy as sp

def log_loss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

def multiclass_log_loss(classes_np, prediction_np):

    classes = classes_np.tolist()
    prediction = prediction_np.tolist()

    class_array = []
    class_count = len(set(classes))
    for i in classes:
        row = [0] * class_count
        row[i] = 1
        class_array.append(row)

    scores = []
    for index in range(0, len(prediction)):
        result = log_loss(class_array[index], prediction[index])
        scores.append(result)

    return sum(scores) / len(scores)



if __name__ == '__main__':
    main()