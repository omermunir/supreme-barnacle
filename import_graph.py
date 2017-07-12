import tensorflow as tf

with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('my-model.meta')
  new_saver.restore(sess, tf.train.latest_checkpoint('./'))









 


