import sys
import numpy as np
import tensorflow as tf


from sklearn.metrics import roc_curve, auc

labels = np.array([0, 0, 1, 1, 0])
probas = np.array([0.1, 0.4, 0.35, 0.8, 0.3])


import keras.backend as K

def k_custom_auc(y_true, y_pred):
    fpr, tpr, dash = roc_curve(y_true, y_pred)
    score = auc(fpr, tpr)
    return score


def get_pred_array(probas):

    scores = np.sort(probas)
    pred_array = []
    for val in scores:
        pred_thr = probas >= val
        pred_thr = pred_thr.astype(int)
        pred_array.append(pred_thr)

    return pred_array

def conf_matrix(y_true, y_pred):
    TP, FP, TN, FN = 0,0,0,0
    for i, true_val in enumerate(y_true):
        if (y_pred[i] == 1 and true_val ==1 ):
            TP +=1
        elif (y_pred[i] == 1 and true_val ==0):
            FP +=1
        elif (y_pred[i] == 0 and true_val ==0):
            TN +=1
        elif (y_pred[i] == 0 and true_val ==1 ):
            FN +=1
        else:
            raise Exception("conf_matrix() error. "
                            "Invalid confusion matrix conditions.\n "
                            "y_true, y_pred must be integers or booleans.")

    TPR = float(TP / (TP + FN))
    FPR = float(FP / (FP + TN))

    return FPR, TPR

pred_array = get_pred_array(probas)
print(pred_array)

print("(FPR, TPR) Tuples")

ROC_POINTS = []
for pred_vec in pred_array:
    ROC_POINTS.append(conf_matrix(labels, pred_vec))

L = []
for k in reversed(ROC_POINTS):
    L.append(k)

ROC_POINTS = L
print(ROC_POINTS)

sum = 0.0
for k in range(1, len(ROC_POINTS)):
    sum += 0.5 * (ROC_POINTS[k][1] + ROC_POINTS[k - 1][1])*(ROC_POINTS[k][0] - ROC_POINTS[k-1][0])

print("BY HAND AUC score:")
print(sum)


#Sum[(y[[k]] + y[[k - 1]])/2 (X[[k]] - X[[k - 1]]), {k, 2, dim}]


print("\nSklearn AUC score:")
print(k_custom_auc(labels,probas))








sys.exit()

def tflow_auc(labels, preds):
    auc, update_op = tf.metrics.auc(labels, preds)
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        #print("tf auc: {}".format(sess.run([auc, update_op])))
        return sess.run([auc, update_op])
if False:
    print(tflow_auc(labels,preds))
    print(type(tflow_auc(labels,preds)))

def k_custom_auc(y_true, y_pred):
    fpr, tpr, dash = roc_curve(y_true, y_pred)
    score = tf.constant(auc(fpr, tpr))
    sess = tf.Session()
    return sess.run(score)


def k_custom_auc2(y_true, y_pred):
    fpr, tpr, dash = roc_curve(y_true, y_pred)
    score = tf.constant(auc(fpr, tpr))

    return score

def k_custom_auc3(y_true, y_pred):
    fpr, tpr, dash = roc_curve(y_true, y_pred)
    score = tf.Variable(auc(fpr, tpr))

    return score

if True:
    print(k_custom_auc(labels,preds))
    print(type(k_custom_auc(labels,preds)))
if False:
    print(k_custom_auc2(labels,preds))
    print(type(k_custom_auc2(labels,preds)))
if False:
    print(k_custom_auc3(labels,preds))
    print(type(k_custom_auc3(labels,preds)))


#Testing tflows as Keras backend
if False:
    import keras.backend as K

    def mean_pred(y_pred):
        y_pred = K.variable(y_pred)
        print(y_pred)
        print(K.get_value(y_pred))
        return K.mean(y_pred)

    #same as getting a tensorflow session
    sess = K.get_session()

    with sess.as_default():
        x = mean_pred(labels)
        y= K.get_value(x)
        print(y)






#Initial tests
if False:

    # Build a graph.
    a = tf.constant(5.0)
    b = tf.constant(6.0)
    c = a * b

    # Launch the graph in a session.
    sess = tf.Session()

    # Evaluate the tensor `c`.
    print(sess.run(c))

    def fooYoung(x,y):
        sess = tf.Session()
        z = x*y
        return sess.run(z)

    print(fooYoung(a,b))

    m = 1.6
    n = 2.0

    j = tf.constant(m*n)
    print(fooYoung(j,j))
