import tensorflow as tf

train_path = "/home/ubuntu/nfs/new_lsa64/lsa_train_data.tfrecords"
test_path  = "/home/ubuntu/nfs/new_lsa64/lsa_test_data.tfrecords"

def tensor_cast(inputs, labels, mask):
    inputs = tf.cast(inputs, tf.float32)
    return inputs, labels-1, mask


#tfrecordの処理の参考URL
#https://www.tensorflow.org/tutorials/load_data/tfrecord
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() 
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(video, label, image_shape, mask):
    feature = {
        'video': _bytes_feature(video),
        'label': _int64_feature(label),
        'len_seq':_int64_feature(image_shape[0]),
        'height': _int64_feature(image_shape[1]),
        'width': _int64_feature(image_shape[2]),
        'depth': _int64_feature(image_shape[3]),
        'mask' : _bytes_feature(mask),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def parse_tfrecord(serialized_example):
    feature_description = {
        'video' : tf.io.FixedLenFeature((), tf.string),
        'label': tf.io.FixedLenFeature((), tf.int64),
        'len_seq':tf.io.FixedLenFeature((), tf.int64),
        'height': tf.io.FixedLenFeature((), tf.int64),
        'width': tf.io.FixedLenFeature((), tf.int64),
        'depth': tf.io.FixedLenFeature((), tf.int64),
        'mask' : tf.io.FixedLenFeature((), tf.string),
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    
    image = tf.io.parse_tensor(example['video'], out_type = tf.float64)
    image_shape = [example['len_seq'], example['height'], example['width'], example['depth']]
    image = tf.reshape(image, image_shape)

    mask = tf.io.parse_tensor(example['mask'], out_type = tf.bool)

    return image, example['label'], mask

def read_dataset(BATCH_SIZE):
    tfrecord_train_dataset = tf.data.TFRecordDataset(train_path)
    parsed_train_dataset = tfrecord_train_dataset.map(parse_tfrecord)
    tfrecord_test_dataset = tf.data.TFRecordDataset(test_path)
    parsed_test_dataset = tfrecord_test_dataset.map(parse_tfrecord)

    padded_shape = (tf.constant(-1.0, dtype=tf.float32), tf.constant(0, dtype=tf.int64), tf.constant(False, dtype=tf.bool))
    train_dataset = parsed_train_dataset.map(tensor_cast).shuffle(buffer_size=10, seed=100).padded_batch(BATCH_SIZE,padded_shapes=([201, 112, 200, 3], [], [201]), padding_values=padded_shape).prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset =  parsed_test_dataset.map(tensor_cast).shuffle(buffer_size=10,  seed=100).padded_batch(BATCH_SIZE,  padded_shapes=([201, 112, 200, 3], [], [201]), padding_values=padded_shape).prefetch(tf.data.experimental.AUTOTUNE)
    return train_dataset, test_dataset