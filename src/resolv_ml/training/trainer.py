# TODO - DOC
import json
import logging
import time
from pathlib import Path
from typing import Union, Callable, Dict

import keras
import math
from keras import callbacks


class Trainer:

    def __init__(self, model: keras.Model, config_file_path: Union[Path, str]):
        with open(config_file_path) as file:
            self._config = json.load(file)
        self._model = model
        if "output_dir" in self._config:
            output_dir_config = self._config['output_dir']
            self._output_dir = output_dir_config.get("path", "./")
            if "timestamp_format" in output_dir_config:
                timestamp = time.strftime(output_dir_config["timestamp_format"])
                self._output_dir += f'/{timestamp}'
        else:
            self._output_dir = ""

    @property
    def config(self):
        return self._config

    def train(self,
              train_data,
              train_data_cardinality=None,
              validation_data=None,
              validation_data_cardinality=None,
              class_weight=None,
              sample_weight=None,
              custom_callbacks: callbacks.Callback = None,
              lr_schedule: Callable = None,
              lr_scheduler_verbose: int = 0,
              lambda_callbacks: Dict[str, Callable] = None):
        if not self._model.compiled:
            raise ValueError("Model is not compiled. Please call compile() before training.")

        fit_config = self._config['fit'].copy()
        batch_size = fit_config['batch_size']

        total_steps = fit_config.pop('total_steps')
        if total_steps:
            fit_config.pop('steps_per_epoch')
            steps_per_epoch = total_steps // fit_config.pop('validation_freq_steps')
            fit_config["epochs"] = total_steps // steps_per_epoch
            if train_data_cardinality:
                max_dataset_steps = train_data_cardinality // batch_size
                if max_dataset_steps < total_steps:
                    logging.warning(f'The given number of total training steps ({total_steps}) exceeds the maximum '
                                    f'number of possible steps ({max_dataset_steps}) for a dataset with cardinality '
                                    f'{train_data_cardinality} and batch size {batch_size}. '
                                    f'Be sure to call repeat() on the dataset.')
        else:
            steps_per_epoch = fit_config.pop('steps_per_epoch')
            if not steps_per_epoch and train_data_cardinality:
                steps_per_epoch = train_data_cardinality // batch_size

        validation_steps = fit_config.pop('validation_steps')
        if not validation_steps and validation_data_cardinality:
            validation_batch_size = fit_config.get('validation_batch_size') or batch_size
            validation_steps = validation_data_cardinality // validation_batch_size

        history = self._model.fit(
            train_data,
            callbacks=self._get_callbacks(custom_callbacks, lr_schedule, lr_scheduler_verbose, lambda_callbacks),
            validation_data=validation_data,
            class_weight=class_weight,
            sample_weight=sample_weight,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            **fit_config
        )
        return history

    def compile(self,
                loss=None,
                metrics=None,
                weighted_metrics=None,
                lr_schedule: keras.optimizers.schedules.LearningRateSchedule = None) -> keras.Model:
        compile_config = self._config['compile']
        if lr_schedule:
            compile_config['optimizer']["config"]['learning_rate'] = keras.optimizers.schedules.serialize(lr_schedule)
        self._model.compile(loss=loss, metrics=metrics, weighted_metrics=weighted_metrics, **compile_config)
        return self._model

    def _get_callbacks(self,
                       custom_callbacks: callbacks.Callback = None,
                       lr_schedule: Callable = None,
                       lr_scheduler_verbose: int = 0,
                       lambda_callbacks: Dict[str, Callable] = None):
        training_callbacks = custom_callbacks or []
        callbacks_config = self._config['callbacks']

        if "model_checkpoint" in callbacks_config:
            config = callbacks_config["model_checkpoint"]
            config["filepath"] = f'{self._output_dir}/{config["filepath"]}'
            model_checkpoint = callbacks.ModelCheckpoint(**config)
            training_callbacks.append(model_checkpoint)
        if "backup_and_restore" in callbacks_config:
            config = callbacks_config["backup_and_restore"]
            config["backup_dir"] = f'{self._output_dir}/{config["backup_dir"]}'
            backup_and_restore = callbacks.BackupAndRestore(**config)
            training_callbacks.append(backup_and_restore)
        if "tensorboard" in callbacks_config:
            config = callbacks_config["tensorboard"]
            config["log_dir"] = f'{self._output_dir}/{config["log_dir"]}'
            tensorboard = callbacks.TensorBoard(**config)
            training_callbacks.append(tensorboard)
        if "early_stopping" in callbacks_config:
            early_stopping = callbacks.EarlyStopping(**callbacks_config["early_stopping"])
            training_callbacks.append(early_stopping)
        if lr_schedule:
            lr_scheduler = callbacks.LearningRateScheduler(schedule=lr_schedule, verbose=lr_scheduler_verbose)
            training_callbacks.append(lr_scheduler)
        if "reduce_lr_on_plateau" in callbacks_config:
            reduce_lr_on_plateau = callbacks.ReduceLROnPlateau(**callbacks_config["reduce_lr_on_plateau"])
            training_callbacks.append(reduce_lr_on_plateau)
        if "remote_monitor" in callbacks_config:
            remote_monitor = callbacks.RemoteMonitor(**callbacks_config["remote_monitor"])
            training_callbacks.append(remote_monitor)
        if lambda_callbacks:
            lambda_callbacks = callbacks.LambdaCallback(**lambda_callbacks)
            training_callbacks.append(lambda_callbacks)
        if callbacks_config.get("terminate_on_nan", False):
            training_callbacks.append(callbacks.TerminateOnNaN())
        if "csv_logger" in callbacks_config:
            config = callbacks_config["csv_logger"]
            config["filename"] = f'{self._output_dir}/{config["filename"]}'
            csv_logger = callbacks.CSVLogger(**config)
            training_callbacks.append(csv_logger)
        if callbacks_config.get("progbar_logger", False):
            training_callbacks.append(callbacks.ProgbarLogger())
        if "swap_ema_weights" in callbacks_config:
            swap_ema_weights = callbacks.SwapEMAWeights(**callbacks_config["swap_ema_weights"])
            training_callbacks.append(swap_ema_weights)
        return training_callbacks
