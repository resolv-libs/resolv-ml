""" TODO - doc"""
import keras


class LearningRateLoggerCallback(keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        logs.update({'lr': self.model.optimizer.learning_rate.numpy()})

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': self.model.optimizer.learning_rate.numpy()})
