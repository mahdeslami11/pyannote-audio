#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2020 CNRS

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

from .base_labeling import BaseLabeling
import torch
from typing import Text
from typing import Dict
from pyannote.database.protocol.protocol import Protocol
from functools import partial
import numpy as np
import scipy.optimize
from pyannote.audio.features import Pretrained


from pyannote.audio.pipeline import SpeechActivityDetection
from .speech_detection import validate_helper_func


class Diarization(BaseLabeling):

    def validation_criterion(self, protocol: Protocol,
                                   **kwargs) -> Text:
        return f'detection_fscore'

    def validate_epoch(self,
                       epoch: int,
                       validation_data,
                       device: torch.device = None,
                       batch_size: int = 32,
                       n_jobs: int = 1,
                       duration: float = None,
                       step: float = 0.25,
                       detection: bool = False,
                       **kwargs) -> Dict:

        # compute (and store) SAD scores
        postprocess = partial(np.mean, axis=2, keepdims=True)
        pretrained = Pretrained(validate_dir=self.validate_dir_,
                                epoch=epoch,
                                duration=duration,
                                step=step,
                                batch_size=batch_size,
                                device=device,
                                postprocess=postprocess)

        for current_file in validation_data:
            current_file['scores'] = pretrained(current_file)

        # pipeline
        pipeline = SpeechActivityDetection(scores="@scores",
                                           fscore=True)

        def fun(threshold):
            pipeline.instantiate({'onset': threshold,
                                  'offset': threshold,
                                  'min_duration_on': 0.100,
                                  'min_duration_off': 0.100,
                                  'pad_onset': 0.,
                                  'pad_offset': 0.})
            metric = pipeline.get_metric(parallel=True)
            validate = partial(validate_helper_func,
                               pipeline=pipeline,
                               metric=metric)
            if n_jobs > 1:
                _ = self.pool_.map(validate, validation_data)
            else:
                for file in validation_data:
                    _ = validate(file)

            return 1. - abs(metric)

        res = scipy.optimize.minimize_scalar(
            fun, bounds=(0., 1.), method='bounded', options={'maxiter': 10})

        threshold = res.x.item()

        return {'metric': self.validation_criterion(None),
                'minimize': False,
                'value': float(1. - res.fun),
                'pipeline': pipeline.instantiate({'onset': threshold,
                                                  'offset': threshold,
                                                  'min_duration_on': 0.100,
                                                  'min_duration_off': 0.100,
                                                  'pad_onset': 0.,
                                                  'pad_offset': 0.})}
