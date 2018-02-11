

import tensorflow as tf

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
