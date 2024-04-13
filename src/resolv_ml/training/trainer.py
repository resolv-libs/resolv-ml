# TODO - DOC
import json
from pathlib import Path
from typing import Union, Callable, Dict

import keras
from keras import callbacks


class Trainer:

    def __init__(self, model: keras.Model, config_file_path: Union[Path, str]):
        with open(config_file_path) as file:
            self._config = json.load(file)
        self._model = model

    def train(self,
              train_data,
              validation_data=None,
              validation_split=0.0,
              class_weight=None,
              sample_weight=None,
              custom_callbacks: callbacks.Callback = None,
              lr_schedule: Callable = None,
              lr_scheduler_verbose: int = 0,
              lambda_callbacks: Dict[str, Callable] = None):
        if not self._model.compiled:
            raise ValueError("Model is not compiled. Please call compile() before training.")
        history = self._model.fit(
            train_data,
            callbacks=self._get_callbacks(custom_callbacks, lr_schedule, lr_scheduler_verbose, lambda_callbacks),
            validation_split=validation_split,
            validation_data=validation_data,
            class_weight=class_weight,
            sample_weight=sample_weight,
            **self._config['fit']
        )
        return history

    def compile(self,
                loss=None,
                metrics=None,
                weighted_metrics=None) -> keras.Model:
        self._model.compile(loss=loss, metrics=metrics, weighted_metrics=weighted_metrics, **self._config['compile'])
        return self._model

    def _get_callbacks(self,
                       custom_callbacks: callbacks.Callback = None,
                       lr_schedule: Callable = None,
                       lr_scheduler_verbose: int = 0,
                       lambda_callbacks: Dict[str, Callable] = None):
        training_callbacks = custom_callbacks if custom_callbacks else []
        callbacks_config = self._config['callbacks']
        if "model_checkpoint" in callbacks_config:
            model_checkpoint = callbacks.ModelCheckpoint(**callbacks_config["model_checkpoint"])
            training_callbacks.append(model_checkpoint)
        if "backup_and_restore" in callbacks_config:
            backup_and_restore = callbacks.BackupAndRestore(**callbacks_config["backup_and_restore"])
            training_callbacks.append(backup_and_restore)
        if "tensorboard" in callbacks_config:
            tensorboard = callbacks.TensorBoard(**callbacks_config["tensorboard"])
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
            csv_logger = callbacks.CSVLogger(**callbacks_config["csv_logger"])
            training_callbacks.append(csv_logger)
        if callbacks_config.get("progbar_logger", False):
            training_callbacks.append(callbacks.ProgbarLogger())
        if "swap_ema_weights" in callbacks_config:
            swap_ema_weights = callbacks.SwapEMAWeights(**callbacks_config["swap_ema_weights"])
            training_callbacks.append(swap_ema_weights)
        return training_callbacks
