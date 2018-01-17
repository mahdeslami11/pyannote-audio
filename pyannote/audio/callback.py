#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016-2017 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr


import matplotlib
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import os.path
import datetime
import numpy as np
from pyannote.audio.util import mkdir_p
from keras.callbacks import Callback


class LoggingCallback(Callback):
    """Logging callback

    Parameters
    ----------
    log_dir : str
    log : list of tuples, optional
        Defaults to [('train', 'loss')]
    extract_embedding : func
        Function that takes Keras model as input and returns the actual model.
        This is useful for embedding that are not directly optimized by Keras.
        Defaults to identity function.
    restart : boolean, optional
        Indicates that this training is a restart, not a cold start (default).
    """

    ARCHITECTURE_YML = '{log_dir}/architecture.yml'
    WEIGHTS_DIR = '{log_dir}/weights'
    WEIGHTS_H5 = '{log_dir}/weights/{epoch:04d}.h5'
    WEIGHTS_PT = '{log_dir}/weights/{epoch:04d}.pt'
    OPTIMIZER_PT = '{log_dir}/weights/{epoch:04d}.optimizer.pt'
    LOG_TXT = '{log_dir}/{name}.{subset}.txt'
    LOG_PNG = '{log_dir}/{name}.{subset}.png'
    LOG_EPS = '{log_dir}/{name}.{subset}.eps'

    def __init__(self, log_dir, log=None, extract_embedding=None,
                 restart=False, backend='keras'):
        super(LoggingCallback, self).__init__()

        # make sure path is absolute
        self.log_dir = os.path.realpath(log_dir)

        if extract_embedding is None:
            extract_embedding = lambda model: model
        self.extract_embedding = extract_embedding

        # create log_dir directory
        mkdir_p(self.log_dir)

        # this will fail if the directory already exists
        # and this is OK  because 'weights' directory
        # usually contains the output of very long computations
        # and you do not want to erase them by mistake :/
        self.restart = restart
        if not self.restart:
            weights_dir = self.WEIGHTS_DIR.format(log_dir=self.log_dir)
            os.makedirs(weights_dir)

        if log is None:
            log = [('train', 'loss')]
        self.log = log

        self.values = {}
        for subset, name in self.log:
            if subset not in self.values:
                self.values[subset] = {}
            if name not in self.values[subset]:
                self.values[subset][name] = []

        self.backend = backend

    def get_loss(self, epoch, subset, logs={}):
        if subset != 'train':
            raise ValueError('"loss" is only available for "train"')
        value = logs['loss']
        minimize = True
        return value, minimize

    def get_accuracy(self, epoch, subset, logs={}):
        if subset != 'train':
            raise ValueError('"accuracy" is only available for "train"')
        value = logs['acc']
        minimize = False
        return value, minimize

    def get_value(self, epoch, name, subset, logs={}):
        get_value = getattr(self, 'get_' + name)
        value, minimize = get_value(epoch, subset, logs=logs)
        return value, minimize

    def on_epoch_end(self, epoch, logs={}):
        """Save weights (and various curves) after each epoch"""

        # keep track of when the epoch ended
        now = datetime.datetime.now().isoformat()

        # save model after this epoch

        if self.backend == 'keras':
            import keras.models

            weights_h5 = self.WEIGHTS_H5.format(log_dir=self.log_dir,
                                                epoch=epoch)
            # overwrite only in case of a restart
            # include optimizer only every 10 epochs
            keras.models.save_model(self.model, weights_h5,
                                    overwrite=self.restart,
                                    include_optimizer=(epoch % 10 == 0))

        elif self.backend == 'pytorch':
            import torch

            weights_pt = self.WEIGHTS_PT.format(
                log_dir=self.log_dir, epoch=epoch)
            torch.save(self.model.state_dict(), weights_pt)

            optimizer_pt = self.OPTIMIZER_PT.format(
                log_dir=self.log_dir, epoch=epoch)
            torch.save(self.optimizer.state_dict(), optimizer_pt)

        for subset, name in self.log:

            value, minimize = self.get_value(epoch, name, subset, logs=logs)

            # keep track of value after last epoch
            self.values[subset][name].append(value)
            values = self.values[subset][name]

            # write value to file
            log_txt = self.LOG_TXT.format(
                log_dir=self.log_dir, name=name, subset=subset)
            TXT_TEMPLATE = '{epoch:d} ' + now + ' {value:.8f}\n'

            mode = 'a' if epoch > 0 else 'w'
            try:
                with open(log_txt, mode) as fp:
                    fp.write(TXT_TEMPLATE.format(epoch=epoch, value=value))
                    fp.flush()
            except Exception as e:
                pass

            # keep track of 'best value' model
            if minimize:
                best_epoch = np.argmin(values)
                best_value = np.min(values)
            else:
                best_epoch = np.argmax(values)
                best_value = np.max(values)

            # plot values to file and mark best value so far
            plt.plot(values, 'b')
            plt.plot([best_epoch], [best_value], 'bo')
            plt.plot([0, epoch], [best_value, best_value], 'k--')
            plt.grid(True)

            plt.xlabel('epoch')

            YLABEL = '{name} on {subset}'
            ylabel = YLABEL.format(name=name, subset=subset)
            plt.ylabel(ylabel)

            TITLE = '{name} = {best_value:.8f} on {subset} @ epoch #{best_epoch:d}'
            title = TITLE.format(name=name, subset=subset, best_value=best_value, best_epoch=best_epoch)
            plt.title(title)

            plt.tight_layout()

            # save plot as PNG
            log_png = self.LOG_PNG.format(
                log_dir=self.log_dir, name=name, subset=subset)
            try:
                plt.savefig(log_png, dpi=75)
            except Exception as e:
                pass

            # save plot as EPS
            log_eps = self.LOG_EPS.format(
                log_dir=self.log_dir, name=name, subset=subset)
            try:
                plt.savefig(log_eps)
            except Exception as e:
                pass

            plt.close()


class BaseLogger(Callback):
    """Callback that accumulates epoch averages of metrics.
    This callback is automatically applied to every Keras model.
    """

    def on_epoch_begin(self, epoch, logs=None):
        self.seen = 0
        self.totals = {}

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        batch_size = logs.get('size', 0)
        self.seen += batch_size

        for k, v in logs.items():
            if k in self.totals:
                try:
                    self.totals[k] += v * batch_size
                except ValueError as e:
                    pass
            else:
                self.totals[k] = v * batch_size

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for k in self.params['metrics']:
                if k in self.totals:
                    # Make value available to next callbacks.
                    logs[k] = self.totals[k] / self.seen

class Debugging(Callback):

    GRADIENT_PNG = '{log_dir}/gradient.png'

    def __init__(self):
        super(Debugging, self).__init__()
        self.min_gradient = []
        self.max_gradient = []
        self.avg_gradient = []
        self.pct_gradient = []

    def on_batch_end(self, batch, logs=None):

        min_gradient = np.min(np.abs(logs['gradient']))
        max_gradient = np.max(np.abs(logs['gradient']))
        avg_gradient = np.average(np.abs(logs['gradient']))

        self.min_gradient.append(min_gradient)
        self.max_gradient.append(max_gradient)
        self.avg_gradient.append(avg_gradient)

        # plot values to file and mark best value so far
        plt.plot(self.min_gradient, 'b', label='min')
        plt.plot(self.max_gradient, 'r', label='max')
        plt.plot(self.avg_gradient, 'g', label='avg')
        plt.legend()
        plt.title('Gradients')
        plt.tight_layout()

        # save plot as PNG
        gradient_png = self.GRADIENT_PNG.format(log_dir=logs['log_dir'])
        try:
            plt.savefig(gradient_png, dpi=75)
        except Exception as e:
            pass

        plt.close()


class LoggingCallbackPytorch(object):
    """Logging callback

    Parameters
    ----------
    log_dir : str
    restart : boolean, optional
        Indicates that this training is a restart, not a cold start (default).
    """

    WEIGHTS_DIR = '{log_dir}/weights'
    WEIGHTS_PT = '{log_dir}/weights/{epoch:04d}.pt'
    OPTIMIZER_PT = '{log_dir}/weights/{epoch:04d}.optimizer.pt'

    def __init__(self, log_dir, restart=False):
        super(LoggingCallbackPytorch, self).__init__()

        # make sure path is absolute
        self.log_dir = os.path.realpath(log_dir)

        # create log_dir directory
        mkdir_p(self.log_dir)

        # this will fail if the directory already exists
        # and this is OK  because 'weights' directory
        # usually contains the output of very long computations
        # and you do not want to erase them by mistake :/
        self.restart = restart
        if not self.restart:
            weights_dir = self.WEIGHTS_DIR.format(log_dir=self.log_dir)
            os.makedirs(weights_dir)

        self.values_ = {}

    def on_epoch_end(self, epoch, logs={}):
        """Save weights after each epoch"""

        # keep track of when the epoch ended
        now = datetime.datetime.now().isoformat()

        # save model after this epoch
        import torch

        weights_pt = self.WEIGHTS_PT.format(
            log_dir=self.log_dir, epoch=epoch)
        torch.save(self.model.state_dict(), weights_pt)

        optimizer_pt = self.OPTIMIZER_PT.format(
            log_dir=self.log_dir, epoch=epoch)
        torch.save(self.optimizer.state_dict(), optimizer_pt)
