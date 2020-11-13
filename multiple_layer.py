    with tf.variable_scope("Conv_Prune", "Conv_Prune", [images]) as scope:
                         

        prune_input = tf.keras.Input(tensor=images)
        conv1 = keras.layers.Conv2D(12, 11, 4)(prune_input)
        pool1 = keras.layers.MaxPool2D(3, 2, padding='SAME')(conv1)
        conv2 = keras.layers.Conv2D(24, 9, 1, padding='SAME')(pool1)
        pool2 = keras.layers.MaxPool2D(3, 2, padding='SAME')(conv2)
        conv3 = keras.layers.Conv2D(48, 7, 1, padding='SAME')(pool2)
        pool3 = keras.layers.MaxPool2D(3, 2, padding='SAME')(conv3)
        conv4 = keras.layers.Conv2D(96, 5, 1, padding='SAME')(pool3)
        pool4 = keras.layers.MaxPool2D(3, 2, padding='SAME')(conv4)
        conv5 = keras.layers.Conv2D(192, 3, 1, padding='SAME')(pool4)
        pool5 = keras.layers.MaxPool2D(3, 2, padding='SAME')(conv5)
        conv6 = keras.layers.Conv2D(384, 1, 1, padding='SAME')(pool5)
        pool6 = keras.layers.MaxPool2D(3, 2, padding='SAME')(conv6)

        flat = keras.layers.Flatten()(pool6)
        full1 = keras.layers.Dense(512)(flat)
        drop1 = keras.layers.Dropout(1-pkeep)(full1)
        full2 = keras.layers.Dense(512)(drop1)
        drop2 = keras.layers.Dropout(1-pkeep)(full2)
        
        model = keras.Model(prune_input, drop2, name="model")

        print('------------------OUTPUT------------------ [%s]' % drop2)

    with tf.variable_scope('output') as scope:
        
        weights = tf.Variable(tf.random_normal([512, nlabels], mean=0.0, stddev=0.01), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[nlabels], dtype=tf.float32), name='biases')
        output = tf.add(tf.matmul(drop2, weights), biases, name=scope.name)

        print('------------------OUTPUT------------------ [%s]' % output)

    return [prune_input, output, model]