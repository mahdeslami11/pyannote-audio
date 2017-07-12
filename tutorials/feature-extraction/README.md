> The MIT License (MIT)
>
> Copyright (c) 2017 CNRS
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.
>
> AUTHOR  
> Hervé Bredin - http://herve.niderb.fr

# Feature extraction with `pyannote.audio`

In this tutorial, you will learn how to perform feature extraction using `pyannote-speech-feature` command line tool.

## Table of contents
- [Installation](#installation)
- [ETAPE database](#etape-database)
- [Configuration](#configuration)
- [Extraction](#extraction)
- [Usage](#usage)
  - [In other command line tools](#in-other-command-line-tools)
  - [In your own code](#in-your-own-code)

## Installation
([↑up to table of contents](#table-of-contents))

```bash
$ conda create --name py35-pyannote-audio python=3.5 anaconda
$ source activate py35-pyannote-audio
$ conda install -c conda-forge yaafe
$ pip install -U pip setuptools
$ pip install pyannote.audio
$ pip install pyannote.db.etape
```

## ETAPE database
([↑up to table of contents](#table-of-contents))

This tutorial relies on the [ETAPE database](http://islrn.org/resources/425-777-374-455-4/). We first need to tell `pyannote` where the audio files are located:

```bash
$ cat ~/.pyannote/db.yml
Etape: /path/to/Etape/corpus/{uri}.wav
```

If you want to train the network using a different database, you might need to create your own [`pyannote.database`](http://github.com/pyannote/pyannote-database) plugin.
See [github.com/pyannote/pyannote-db-template](https://github.com/pyannote/pyannote-db-template) for details on how to do so.

## Configuration
([↑up to table of contents](#table-of-contents))

To ensure reproducibility, `pyannote-speech-feature` relies on a configuration file defining the experimental setup:

```bash
$ cat tutorials/feature-extraction/config.yml
feature_extraction:
   name: YaafeMFCC               # extract MFCC using Yaafe
   params:
      coefs: 11                  # 11 coefs
      D: True                    # with coefs 1st derivative
      DD: True                   # with coefs 2nd derivative
      e: False                   # without energy
      De: True                   # with energy 1st derivative
      DDe: True                  # with energy 2nd derivative
      step: 0.010                # every 10ms
      duration: 0.020            # using 20ms-long windows
```

## Extraction
([↑up to table of contents](#table-of-contents))

The following command will extract features for all files the `TV` protocol of the ETAPE database.

```bash
$ export EXPERIMENT_DIR=tutorials/feature-extraction
$ pyannote-speech-feature ${EXPERIMENT_DIR} Etape.SpeakerDiarization.TV
Training set: 28it [01:35,  2.16s/it]
Development set: 9it [00:30,  2.93s/it]
Test set: 9it [00:28,  2.89s/it]
```

This will create a bunch of files in `EXPERIMENT_DIR`.

## Usage
([↑up to table of contents](#table-of-contents))

### In other command line tools
([↑up to table of contents](#table-of-contents))

Now that features are extracted, they can be used by other command line tools (instead of re-computing them on-the-fly).

For instance, the `feature_extraction` section of the configuration file of the [speech activity detection tutorial](/tutorials/speech-activity-detection) can be updated to look like that:

```bash
$ cat tutorials/speech-activity-detection/config.yml
feature_extraction:
   name: Precomputed
   params:
      root_dir: tutorials/feature-extraction

[...]
```

### In your own code
([↑up to table of contents](#table-of-contents))

```python
>>> from pyannote.audio.features import Precomputed
>>> precomputed = Precomputed('tutorials/feature-extraction')
>>> from pyannote.database import get_protocol
>>> protocol = get_protocol('Etape.SpeakerDiarization.TV')
>>> for current_file in protocol.test():
...     features = precomputed(current_file)
...     break
>>> X = features.data                  # numpy array containing all features
>>> from pyannote.core import Segment
>>> X.crop(Segment(10.2, 11.4))        # numpy array containing subset of features
```

## Going further...
([↑up to table of contents](#table-of-contents))

```bash
$ pyannote-speech-feature --help
```
