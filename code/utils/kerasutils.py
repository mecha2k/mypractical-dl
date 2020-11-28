import tensorflow as tf


def init_gpus():
    tf.debugging.set_log_device_placement(True)

    physical_gpus = tf.config.experimental.list_physical_devices("GPU")
    if physical_gpus:
        try:
            for gpu in physical_gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(physical_gpus), "Physical GPUs, ", len(logical_gpus))
        except RuntimeError as e:
            print(e)
