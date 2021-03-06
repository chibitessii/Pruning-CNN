    weight_decay = 0.0005
    weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    
    with tf.variable_scope("Layer_Prune", "Layer_Prune", [images]) as scope:
                         

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

        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                               final_sparsity=0.80,
                                                               begin_step=0,
                                                               end_step=100)
        }

        pruned_model = keras.Model(prune_input, drop2, name="model_for_pruning")
        model_for_pruning = prune_low_magnitude(pruned_model, **pruning_params)

        layer_input = model_for_pruning.input                                         
        #prune_output = [model_for_pruning.output for drop2 in model_for_pruning.layers] 
        prune_output = model_for_pruning.output 

        print('-----------------DROP2------------------ [%s]' % model_for_pruning.output)
