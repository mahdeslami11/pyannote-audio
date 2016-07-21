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
# Hervé BREDIN - http://herve.niderb.fr

"""
Speaker change detection

Usage:
  speaker_change_detection apply [options] <embedding_dir> <input.wav> <output.json>
  speaker_change_detection tune <embedding_dir>
  speaker_change_detection -h | --help
  speaker_change_detection --version

Options:
  <embedding_dir>           Path to embedding directory
  <input.wav>               Path to audio file
  <output.json>             Path to output file (JSON export of pyannote Timeline)
  --precision=<seconds>     Step size in seconds [default: 0.010]
  --min-duration=<seconds>  Minimum duration between two changes [default: 1.0]
  --alpha=<float>           [default: 1.0]
  -h --help                 Show this screen.
  --version                 Show version.
"""

import yaml
from docopt import docopt
from pyannote.core.json import dump_to
from pyannote.audio.embedding.models import SequenceEmbedding
from pyannote.audio.features.yaafe import YaafeMFCC
from pyannote.audio.algorithms.segmentation import EmbeddingSegmenter


if __name__ == '__main__':

    arguments = docopt(__doc__, version='Speaker change detection')

    embedding_dir = arguments['<embedding_dir>']
    precision = float(arguments['--precision'])
    min_duration = float(arguments['--min-duration'])
    alpha = float(arguments['--alpha'])

    segmenter = EmbeddingSegmenter.from_disk(embedding_dir,
                                             precision=precision,
                                             min_duration=min_duration,
                                             alpha=alpha)

    input_wav = arguments['<input.wav>']
    segmentation = segmenter.apply(input_wav)

    output_json = arguments['<output.json>']
    dump_to(segmentation, output_json)
