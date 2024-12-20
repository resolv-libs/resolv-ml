import logging
from typing import List

import tensorflow as tf


def check_tf_gpu_availability():
    logging.info(f'Tensorflow version: {tf.__version__}')
    gpu_list = tf.config.list_physical_devices('GPU')
    if len(gpu_list) > 0:
        logging.info(f'Num GPUs Available: {len(gpu_list)}. List: {gpu_list}')
    return gpu_list


def set_visible_devices(gpu_ids: List[int] = None, memory_growth: bool = False):
    gpu_list = check_tf_gpu_availability()

    if not gpu_list:
        raise SystemExit("GPU not available.")

    if gpu_ids:
        selected_gpus = [gpu_list[gpu] for gpu in gpu_ids]
        logging.info(f"Using provided GPU device {selected_gpus}.")
    else:
        logging.info(f"No GPU ids provided. Using default GPU device {gpu_list[0]}.")
        selected_gpus = [gpu_list[0]]

    tf.config.set_visible_devices(selected_gpus, 'GPU')

    for gpu in selected_gpus:
        logging.info(f"Setting GPU device {gpu} memory growth to {memory_growth}.")
        tf.config.experimental.set_memory_growth(gpu, memory_growth)

    return selected_gpus


def get_distributed_strategy(gpu_ids: List[int] = None, memory_growth: bool = False) -> tf.distribute.Strategy:
    selected_gpus = set_visible_devices(gpu_ids, memory_growth)
    selected_gpus_name = [selected_gpu.name.replace("/physical_device:", "") for selected_gpu in selected_gpus]
    if len(selected_gpus) > 1:
        logging.info(f"Using MirroredStrategy on selected GPUs: {selected_gpus_name}")
        strategy = tf.distribute.MirroredStrategy(devices=selected_gpus_name)
    else:
        logging.info(f"Using OneDeviceStrategy on selected GPU: {selected_gpus_name[0]}")
        strategy = tf.distribute.OneDeviceStrategy(device=selected_gpus_name[0])
    return strategy
