#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016 CNRS

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
# Gregory GELLY - gregory.gelly@limsi.fr


from keras.legacy import interfaces
from keras.optimizers import Optimizer
import keras.backend as K
from pyannote.audio.keras_utils import register_custom_object


class SMORMS3(Optimizer):
    """SMORMS3 optimizer.

    Default parameters follow those provided in the blog post.

    # Arguments
        lr: float >= 0. Learning rate.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.

    # References
        - [RMSprop loses to SMORMS3 - Beware the Epsilon!](http://sifter.org/~simon/journal/20150420.html)
    """

    def __init__(self, lr=0.001, epsilon=1e-16, decay=0.,
                 **kwargs):
        super(SMORMS3, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.lr = K.variable(lr, name='lr')
            self.decay = K.variable(decay, name='decay')
            self.iterations = K.variable(0, dtype='int64', name='iterations')
        self.epsilon = epsilon
        self.initial_decay = decay

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        shapes = [K.get_variable_shape(p) for p in params]
        ms = [K.zeros(shape) for shape in shapes]
        vs = [K.zeros(shape) for shape in shapes]
        mems = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + ms + vs + mems
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))


        for p, g, m, v, mem in zip(params, grads, ms, vs, mems):

            r = 1. / (1. + mem)
            new_m = (1. - r) * m + r * g
            new_v = (1. - r) * v + r * K.square(g)
            denoise = K.square(new_m) / (new_v + self.epsilon)
            new_p = p - g * K.minimum(lr, denoise) / (K.sqrt(new_v) + self.epsilon)
            new_mem = 1. + mem * (1. - denoise)

            self.updates.append(K.update(m, new_m))
            self.updates.append(K.update(v, new_v))
            self.updates.append(K.update(mem, new_mem))

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon}
        base_config = super(SMORMS3, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# register user-defined Keras optimizer
register_custom_object('SMORMS3', SMORMS3)


class SSMORMS3(Optimizer):
    """Stabilized SMORMS3 optimizer.

    Slight variation over SMORMS3 default implementation.

    # Arguments
        epsilon: float >= 0. Fuzz factor.

    # References
        - [RMSprop loses to SMORMS3 - Beware the Epsilon!](http://sifter.org/~simon/journal/20150420.html)
    """

    def __init__(self, epsilon=1e-8, **kwargs):
        super(SSMORMS3, self).__init__(**kwargs)

        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
        self.epsilon = epsilon


    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        shapes = [K.get_variable_shape(p) for p in params]
        ms = [K.zeros(shape) for shape in shapes]
        vs = [K.zeros(shape) for shape in shapes]
        mems = [K.zeros(shape) for shape in shapes]
        denoises = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + ms + vs + mems + denoises
        self.updates = [K.update_add(self.iterations, 1)]

        for p, g, m, v, mem, denoise in zip(params, grads, ms, vs, mems, denoises):

            r = K.minimum(0.2, K.maximum(0.005, 1. / (1. + mem)))
            new_mem = 1. / r - 1.
            new_m = (1. - r) * m + r * g
            new_v = (1. - r) * v + r * K.square(g)
            new_denoise = 0.99 * denoise + 0.01 * K.square(new_m) / (new_v + self.epsilon)
            new_p = p - g * new_denoise / (K.sqrt(new_v) + self.epsilon)
            new_mem = K.maximum(0., 1. + new_mem * (1. - new_denoise))

            self.updates.append(K.update(m, new_m))
            self.updates.append(K.update(v, new_v))
            self.updates.append(K.update(mem, new_mem))
            self.updates.append(K.update(denoise, new_denoise))

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'epsilon': self.epsilon}
        base_config = super(SSMORMS3, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# register user-defined Keras optimizer
register_custom_object('SSMORMS3', SSMORMS3)
