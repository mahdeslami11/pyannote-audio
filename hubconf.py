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

dependencies = ['pyannote.audio']

import os
import yaml
import torch
import zipfile
import shutil
from pathlib import Path
from typing import Optional

from pyannote.audio.utils.path import mkdir_p
from pyannote.audio.labeling.extraction import SequenceLabeling
from pyannote.audio.applications.speech_detection import SpeechActivityDetection


URL = "https://github.com/pyannote/pyannote-audio/releases/download"
RELEASE = "2.0.0-wip"

MODELS = {
    "speech_activity_detection": {
        "AMI": {
            "url": f"{URL}/{RELEASE}/SAD.AMI.d534ec1eb2.zip",
            "hash_prefix": "d534ec1eb2",
        },
        "ETAPE": {
            "url": f"{URL}/{RELEASE}/SAD.ETAPE.bc770a4290.zip",
            "hash_prefix": "bc770a4290",
        },
        "DIHARD": {
            "url": f"{URL}/{RELEASE}/SAD.DIHARD.0585a5507a.zip",
            "hash_prefix": "0585a5507a",
        },
    },
}




def speech_activity_detection(version: str = "AMI",
                              device: Optional[str] = None,
                              batch_size: int = 32,
                              force_reload: bool = False,
                              return_path: bool = False) -> SequenceLabeling:
    """Load pretrained speech activity detection model

    Parameters
    ----------
    version : str
        One of "AMI", "DIHARD", and "ETAPE".
    device : torch.device, optional
        Device used for inference.
    batch_size : int, optional
        Batch size used for inference.
    force_reload : bool
        Whether to discard the existing cache and force a fresh download.
        Defaults to use existing cache.
    return_path : bool
        Return path to model instead of `SequenceLabeling` instance

    Returns
    -------
    model : `SequenceLabeling`

    Usage
    -----
    >>> model = torch.hub.load('pyannote/pyannote-audio',
                               'speech_activity_detection',
                               version='AMI', device='cuda')
    >>> scores = model({'audio': '/path/to/audio.wav'})
    """

    hash_prefix = MODELS['speech_activity_detection'][version]['hash_prefix']
    url = MODELS['speech_activity_detection'][version]['url']

    # path where pre-trained model is downloaded
    hub_dir = Path(os.environ.get("PYANNOTE_AUDIO_HUB",
                                  "~/.pyannote/hub"))
    hub_dir = hub_dir.expanduser().resolve() / hash_prefix

    if not hub_dir.exists() or force_reload:

        if hub_dir.exists():
            shutil.rmtree(hub_dir)

        hub_zip = hub_dir / f"{hash_prefix}.zip"
        mkdir_p(hub_zip.parent)
        try:
            msg = f'Downloading "{url}" to {hub_zip}'
            print(msg)
            torch.hub.download_url_to_file(url, hub_zip,
                                           hash_prefix=hash_prefix,
                                           progress=True)
        except RuntimeError as e:
            shutil.rmtree(hub_dir)
            msg = f'Failed to download model. Please try again.'
            raise RuntimeError(msg)

        # unzip downloaded file
        with zipfile.ZipFile(hub_zip) as z:
            z.extractall(path=hub_dir)

    # content of {hub_dir} directory
    # config.yml
    # train/{training_set}/weights/specs.yml
    # train/{training_set}/weights/{epoch}*.pt
    # train/{training_set}/validate/{development_set}/params.yml

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

    app = SpeechActivityDetection.from_validate_dir(params_yml.parent,
                                                    training=False)

    feature_extraction = app.feature_extraction_
    model = app.model_
    duration = app.task_.duration
    step = 0.25 * duration
    device = torch.device('cpu') if device is None else torch.device(device)

    # initialize embedding extraction
    labeling = SequenceLabeling(
        feature_extraction=feature_extraction,
        model=model,
        duration=duration, step=step,
        batch_size=batch_size, device=device,
        return_intermediate=None)

    return labeling

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
