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

## Installation

```bash
$ conda create --name py35-pyannote-audio python=3.5 anaconda
$ source activate py35-pyannote-audio
$ conda install gcc
$ conda install -c conda-forge yaafe
$ pip install "pyannote.audio==0.2.1"
$ pip install pyannote.db.etape
```

## ETAPE database

This tutorial relies on the [ETAPE database](http://islrn.org/resources/425-777-374-455-4/). We first need to tell `pyannote` where the audio files are located:

```bash
$ cat ~/.pyannote/db.yml
Etape: /path/to/Etape/corpus/{uri}.wav
```

## Configuration

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

The following command will extract features for all files the `TV` protocol of the ETAPE database.

```bash
$ export EXPERIMENT_DIR=tutorials/feature-extraction
$ pyannote-speech-feature ${EXPERIMENT_DIR} Etape.SpeakerDiarization.TV
Training set: 28it [01:35,  2.16s/it]
Development set: 9it [00:30,  2.93s/it]
Test set: 9it [00:28,  2.89s/it]
```

This will create a bunch of files in `EXPERIMENT_DIR`.

## Testing

Now that features are extracted, they can be used by other command line tools (instead of re-computing them on-the-fly).

For instance, the configuration file of the [speech activity detection tutorial](/tutorials/speech-activity-detection) can be updated to look like that:

```bash
$ cat tutorials/speech-activity-detection/config.yml
feature_extraction:
   name: Precomputed
   params:
      root_dir: tutorials/feature-extraction

architecture:
   name: StackedLSTM
   params:
     n_classes: 2
     lstm: [16]
     mlp: [16]
     bidirectional: concat

sequences:
   duration: 3.2
   step: 0.8
   batch_size: 1024
```

## Going further...

```bash
$ pyannote-speech-feature --help
```
