
import numpy as np
import tensorflow as tf


from sklearn.metrics import roc_curve, auc

labels = np.array([0, 0, 1, 1])
preds = np.array([0.1, 0.4, 0.35, 0.8])

def tflow_auc(labels, preds):
    auc, update_op = tf.metrics.auc(labels, preds)
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        #print("tf auc: {}".format(sess.run([auc, update_op])))
        return sess.run([auc, update_op])

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


print(k_custom_auc(labels,preds))
print(type(k_custom_auc(labels,preds)))

print(k_custom_auc2(labels,preds))
print(type(k_custom_auc2(labels,preds)))

print(k_custom_auc3(labels,preds))
print(type(k_custom_auc3(labels,preds)))


import keras.backend as K

def mean_pred(y_pred):
    y_pred = K.variable(y_pred)
    return K.mean(y_pred)

sess = K.get_session()
with sess.as_default():
    print(mean_pred(labels))






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
