#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2019 CNRS

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

dependencies = ['pyannote.audio', 'torch']

import yaml
import pathlib
import typing
import functools

import torch
from pyannote.audio.labeling.extraction import SequenceLabeling \
    as _SequenceLabeling
from pyannote.audio.embedding.extraction import SequenceEmbedding \
    as _SequenceEmbedding
from pyannote.audio.applications.speech_detection import SpeechActivityDetection \
    as _SpeechActivityDetection

MODELS = {
    # speech activity detection
    "sad": {
        "ami": "d534ec1eb2",
        "etape": "bc770a4290",
        "dihard": "0585a5507a",
    },

    # speaker change detection
    "scd": {

    },

    # overlapped speech detection
    "ovl": {

    },

    # speaker embedding
    "emb": {

    },
}

_DEVICE = typing.Union[str, torch.device]
_PRETRAINED = typing.Union[_SequenceLabeling, _SequenceEmbedding, pathlib.Path]

def _generic(task: str = 'sad',
             corpus: str = 'AMI',
             device: typing.Optional[_DEVICE] = None,
             batch_size: int = 32,
             return_path: bool = False) -> _PRETRAINED:
    """Load pretrained model

    Parameters
    ----------
    task : {'sad', 'scd', 'ovl', 'emb'}, optional
        Use 'sad' for speech activity detection, 'scd' for speaker change
        detection, 'ovl' for overlapped speech detection, and 'emb' for speaker
        embedding. Defaults to 'sad'.
    corpus : {'ami', 'dihard', 'etape', 'voxceleb'}, optional
        Use 'ami' for model trained on AMI corpus, 'dihard' for DIHARD corpus,
        'etape' for ETAPE corpus, 'voxceleb' for VoxCeleb corpus.
    device : str or torch.device, optional
        Device used for inference.
    batch_size : int, optional
        Batch size used for inference.
    return_path : bool
        Return path to model checkpoint.
        Defaults to returning `Sequence{Labeling|Embedding}` instance.

    Returns
    -------
    model : `SequenceLabeling` or `SequenceEmbedding` or `Path`

    Usage
    -----
    >>> model = torch.hub.load('pyannote/pyannote-audio', '_generic',
    ...                        task='sad', corpus='ami',
    ...                        device='cuda', batch_size=32)
    >>> sad_scores = model({'audio': '/path/to/audio.wav'})
    """

    # path where pre-trained model is downloaded by torch.hub
    hub_dir = pathlib.Path(__file__).parent / 'models' / task / corpus / MODELS[task][corpus]

    # guess path to "params.yml"
    params_yml, = hub_dir.glob('*/*/*/*/params.yml')

    if return_path:
        # get epoch from params_yml
        with open(params_yml, 'r') as fp:
            params = yaml.load(fp, Loader=yaml.SafeLoader)
            epoch = params['epoch']

        # infer path to model
        return params_yml.parents[2] / 'weights' / f'{epoch:04d}.pt'

    # TODO: print a message to the user providing information about it
    # *_, train, _, development, _ = params_yml.parts
    # msg = 'Model trained on {train}'
    # print(msg)

    if task == 'sad':
        Application = _SpeechActivityDetection
        Extraction = _SequenceLabeling
    else:
        msg = 'Only speech activity detection models (sad) are available.'
        raise ValueError(msg)

    app = Application.from_validate_dir(params_yml.parent,
                                        training=False)
    feature_extraction = app.feature_extraction_
    model = app.model_
    duration = app.task_.duration
    step = 0.25 * duration
    device = torch.device('cpu') if device is None else torch.device(device)

    # initialize  extraction
    return Extraction(
        feature_extraction=feature_extraction,
        model=model,
        duration=duration, step=step,
        batch_size=batch_size, device=device)

_sad = functools.partial(_generic, task='sad')
sad_ami = functools.partial(_sad, corpus='ami')
sad_dihard = functools.partial(_sad, corpus='dihard')
sad_etape = functools.partial(_sad, corpus='etape')

if __name__ == '__main__':
    DOCOPT = """Create torch.hub zip file from validation directory

Usage:
  hubconf.py <validate_dir>
  hubconf.py (-h | --help)
  hubconf.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.
    """
    from docopt import docopt
    from pyannote.audio.applications.base import create_zip
    arguments = docopt(DOCOPT, version='hubconf')
    validate_dir = Path(arguments['<validate_dir>'])
    hub_zip = create_zip(validate_dir)
    print(f'Created file "{hub_zip.name}" in directory "{validate_dir}".')
