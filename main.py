import numpy as np
import tensorflow as tf
import in_data
import tuning.py

gpus = tf.config.experimental.list_physical_devices('GPU')
#GPUのメモリを制限する。
if gpus:
  try:
    for gpu in gpus:
        print(gpu)
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)

#分散学習用インスタンス
mirrored_strategy = tf.distribute.MirroredStrategy()
if __name__ == '__main__':
    tuning.search()