
import tensorflow as tf
from tensorflow.contrib.model_pruning.python import pruning
from tensorflow.contrib.model_pruning.python.layers import layers
from tensorflow.examples.tutorials.mnist import input_data

epochs = 200
batch_size = 40000
model_path_unpruned = "Model_Saves/Unpruned.ckpt"
model_path_pruned = "Model_Saves/Pruned.ckpt"

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
batches = int(len(mnist.train.images) / batch_size)

image = tf.placeholder(tf.float32, [None, 784])
label = tf.placeholder(tf.float32, [None, 10])

# Define the model
layer1 = layers.masked_fully_connected(image, 300)
layer2 = layers.masked_fully_connected(layer1, 300)
logits = layers.masked_fully_connected(layer2, 10)

global_step = tf.train.get_or_create_global_step()
reset_global_step_op = tf.assign(global_step, 0)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=label))

train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step=global_step)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

pruning_hparams = pruning.get_pruning_hparams()
print("Pruning Hyperparameters:", pruning_hparams)

pruning_hparams.begin_pruning_step = 0
pruning_hparams.end_pruning_step = 250
pruning_hparams.pruning_frequency = 1
pruning_hparams.sparsity_function_end_step = 250
pruning_hparams.target_sparsity = .9

p = pruning.Pruning(pruning_hparams, global_step=global_step, sparsity=.9)
prune_op = p.conditional_mask_update_op()

saver = tf.train.Saver()

with tf.Session() as sess:

    sess.run(tf.initialize_all_variables())

    for epoch in range(epochs):
        for batch in range(batches):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_op, feed_dict={image: batch_xs, label: batch_ys})

        if epoch % 10 == 0:
            acc_print = sess.run(accuracy, feed_dict={image: mnist.test.images, label: mnist.test.labels})
            print("Un-pruned model step %d test accuracy %g" % (epoch, acc_print))

    acc_print = sess.run(accuracy, feed_dict={image: mnist.test.images, label: mnist.test.labels})
    print("Pre-Pruning accuracy:", acc_print)
    print("Sparsity of layers (should be 0)", sess.run(tf.contrib.model_pruning.get_weight_sparsity()))

    saver.save(sess, model_path_unpruned)

    sess.run(tf.initialize_all_variables())
    saver.restore(sess, model_path_unpruned)

    sess.run(reset_global_step_op)
    for epoch in range(epochs):
        for batch in range(batches):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Prune and retrain
            sess.run(prune_op)
            sess.run(train_op, feed_dict={image: batch_xs, label: batch_ys})

        if epoch % 10 == 0:
            acc_print = sess.run(accuracy, feed_dict={image: mnist.test.images, label: mnist.test.labels})
            print("Pruned model step %d test accuracy %g" % (epoch, acc_print))
            print("Weight sparsities:", sess.run(tf.contrib.model_pruning.get_weight_sparsity()))

    saver.save(sess, model_path_pruned)
