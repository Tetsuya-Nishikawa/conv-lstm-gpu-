import tensorflow as tf

def zero_padding(inputs, pad):
    padding = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])
    padded_inputs = tf.pad(inputs, padding)
    return padded_inputs
