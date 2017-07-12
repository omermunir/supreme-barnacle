import tensorflow as tf


saver = tf.train.Saver()
# Remember the training_op we want to run by adding it to a collection.
tf.add_to_collection('train_op', train_op)
sess = tf.Session()
for step in xrange(1000000):
    sess.run(train_op)
    if step % 1000 == 0:
        # Saves checkpoint, which by default also exports a meta_graph
        # named 'my-model-global_step.meta'.
        saver.save(sess, 'my-model', global_step=step)

with tf.Session() as sess:
  saver.restore(sess, "/tmp/model.ckpt")
  new_saver = tf.train.import_meta_graph('tmp/my-model.meta')
  new_saver.restore(sess, 'tmp/my-model')
  # tf.get_collection() returns a list. In this example we only want the
  # first one.
  train_op = tf.get_collection('train_op')[0]
  for step in xrange(1000000):
    sess.run(train_op)

