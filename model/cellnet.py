def cellnet(x, num_labels, is_training, verbose=False):


    print("+---------------------------------------------------------------------------------------+")
    print("|                             Building the model -- CellNet                             |")
    print("+---------------------------------------------------------------------------------------+")
    print("|                                                                                       |")
    with tf.compat.v1.variable_scope("cellnet"):
        with tf.compat.v1.variable_scope("conv1"):
            x = conv2d(x, in_channels=1, out_channels=32, kernel_size=11, stride=2, padding="SAME", verbose=verbose)
            x = maxpooling(x, poolsize=2, stride=2, padding="SAME", verbose=verbose)
            x = relu(x, verbose=verbose)
            x = batchnorm(x, is_training, verbose=verbose)

        with tf.compat.v1.variable_scope("conv2"):
            x = conv2d(x, in_channels=32, out_channels=64, kernel_size=6, stride=2, padding="SAME", verbose=verbose)
            x = maxpooling(x, poolsize=2, stride=2, padding="SAME", verbose=verbose)
            x = relu(x, verbose=verbose)
            x = batchnorm(x, is_training, verbose=verbose)

        with tf.compat.v1.variable_scope("conv3"):
            x = conv2d(x, in_channels=64, out_channels=128, kernel_size=3, stride=1, padding="SAME", verbose=verbose)
            x = maxpooling(x, poolsize=2, stride=2, padding="SAME", verbose=verbose)
            x = relu(x, verbose=verbose)
            x = batchnorm(x, is_training, verbose=verbose)

        with tf.compat.v1.variable_scope("fc1"):
            x = flatten(x)
            x = fullyconnected(x, num_out=2048, verbose=verbose)
            x = relu(x, verbose=verbose)

        with tf.compat.v1.variable_scope("fc2"):
            x = fullyconnected(x, num_out=64, verbose=verbose)
            x = relu(x, verbose=verbose)

        with tf.compat.v1.variable_scope("softmax"):
            x = fullyconnected(x, num_out=num_labels, verbose=verbose)
            x = softmax(x, verbose=verbose)
    print("+---------------------------------------------------------------------------------------+")
    print("|                                   Model established                                   |")
    print("+---------------------------------------------------------------------------------------+")

    return x

