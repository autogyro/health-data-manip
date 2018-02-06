
import numpy as np
import tensorflow as tf

labels = np.array([0, 0, 1, 1])
preds = np.array([0.1, 0.4, 0.35, 0.8])

import tensorflow as tf
auc, update_op = tf.metrics.auc(labels, preds)
with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    print("tf auc: {}".format(sess.run([auc, update_op])))



