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

# Speech activity detection with `pyannote.audio`

In this tutorial, you will learn how to train, tune, and test a speech activity detector based on MFCCs and LSTMs, using `pyannote-speech-detection` command line tool.

## Table of contents
- [Citation](#citation)
- [Installation](#installation)
- [Experimental setup](#experimental-setup)
  - [ETAPE database](#etape-database)
  - [Configuration](#configuration)
  - [Training](#training)
  - [Validation](#validation)
  - [Tuning](#tuning)
  - [Testing](#testing)
  - [Evaluation](#evaluation)


## Citation
([↑up to table of contents](#table-of-contents))

If you use `pyannote-audio` for speech activity detection, please cite the following paper:

```bibtex
@inproceedings{Yin2017,
  Author = {Ruiqing Yin and Herv\'e Bredin and Claude Barras},
  Title = {{Speaker Change Detection in Broadcast TV using Bidirectional Long Short-Term Memory Networks}},
  Booktitle = {{Interspeech 2017, 18th Annual Conference of the International Speech Communication Association}},
  Year = {2017},
  Month = {August},
  Address = {Stockholm, Sweden},
  Url = {https://github.com/yinruiqing/change_detection}
}
```

## Installation
([↑up to table of contents](#table-of-contents))

```bash
$ conda create --name py35-pyannote-audio python=3.5 anaconda
$ source activate py35-pyannote-audio
$ conda install -c conda-forge yaafe
$ pip install -U pip setuptools
$ pip install pyannote.audio
$ pip install tensorflow   # or tensorflow-gpu
$ pip install pyannote.db.etape
```

## Experimental setup
([↑up to table of contents](#table-of-contents))

### ETAPE database
([↑up to table of contents](#table-of-contents))

This tutorial relies on the [ETAPE database](http://islrn.org/resources/425-777-374-455-4/). We first need to tell `pyannote` where the audio files are located:

```bash
$ cat ~/.pyannote/db.yml
Etape: /path/to/Etape/corpus/{uri}.wav
```

If you want to train the network using a different database, you might need to create your own [`pyannote.database`](http://github.com/pyannote/pyannote-database) plugin.
See [github.com/pyannote/pyannote-db-template](https://github.com/pyannote/pyannote-db-template) for details on how to do so.

### Configuration
([↑up to table of contents](#table-of-contents))

To ensure reproducibility, `pyannote-speech-detection` relies on a configuration file defining the experimental setup:

```bash
$ cat tutorials/speech-activity-detection/config.yml
feature_extraction:
   name: YaafeMFCC
   params:
      e: False                   # this experiments relies
      De: True                   # on 11 MFCC coefficients
      DDe: True                  # with 1st and 2nd derivatives
      D: True                    # without energy, but with
      DD: True                   # energy derivatives

architecture:
   name: StackedLSTM
   params:                       # this experiments relies
     n_classes: 2                # on one LSTM layer (16 outputs)
     lstm: [16]                  # and one dense layer.
     mlp: [16]                   # LSTM is bidirectional
     bidirectional: concat

sequences:
   duration: 3.2                 # this experiments relies
   step: 0.8                     # on sliding windows of 3.2s
                                 # with a step of 0.8s
```

### Training
([↑up to table of contents](#table-of-contents))

The following command will train the network using the training set of the `TV` protocol of the ETAPE database. This may take a long time...

```bash
$ export EXPERIMENT_DIR=tutorials/speech-activity-detection
$ pyannote-speech-detection train \       #  
          ${EXPERIMENT_DIR} \             # <experiment_dir>
          Etape.SpeakerDiarization.TV     # <database.task.protocol>
Epoch 1/1000
62464/62464 [==============================] - 97s - loss: 0.2268 - acc: 0.9339
Epoch 2/1000
62464/62464 [==============================] - 84s - loss: 0.1570 - acc: 0.9488
...
Epoch 50/1000
62464/62464 [==============================] - 83s - loss: 0.0987 - acc: 0.9687...
```

This will create a bunch of files in `TRAIN_DIR` (defined below), including plots showing the accuracy epoch after epoch.

In the rest of this tutorial, we assume that we killed training after epoch #50.

### Validation
([↑up to table of contents](#table-of-contents))

To get a quick idea of how the network is doing during training, one can use the "validate" mode.
It can (should!) be run in parallel to training and evaluates the model epoch after epoch.

```bash
$ export TRAIN_DIR=${EXPERIMENT_DIR}/train/Etape.SpeakerDiarization.TV.train
$ pyannote-speech-detection validate \
          ${TRAIN_DIR} \               # <train_dir>
          Etape.SpeakerDiarization.TV  # <database.task.protocol>
```

This will create a bunch of files in `TRAIN_DIR/validate`, including plots showing the evolution of detection error rate epoch after epoch.

### Tuning
([↑up to table of contents](#table-of-contents))

Now that the network is trained, we need to tune a bunch of hyper-parameters (including which epoch to use, and [various decision thresholds](https://github.com/pyannote/pyannote-audio/blob/8aaffd98434539304ac52d64097eec9a61bc71ee/pyannote/audio/signal.py#L137-L145))...

This is done on the developement set of the `TV` protocol of the ETAPE database. This may also take a long time...

```bash
$ export TRAIN_DIR=${EXPERIMENT_DIR}/train/Etape.SpeakerDiarization.TV.train
$ pyannote-speech-detection tune \
          ${TRAIN_DIR} \               # <train_dir>
          Etape.SpeakerDiarization.TV  # <database.task.protocol>
```

This will create a `tune.yml` file in `TUNE_DIR` (defined below) containing the best set of hyper-parameters:

```
$ cat ${TUNE_DIR}/tune.yml
status:
  epochs: 50                         # 50 epochs were available for tuning
  objective: 0.044285906870045855    # best detection error rate
epoch: 49                            # best set
offset: 0.42448453210770987          # of hyper-
onset: 0.661734768362515             # parameters
```

### Testing
([↑up to table of contents](#table-of-contents))

Now that the speech activity detector is trained and tuned, we can apply it on the test set of the `TV` protocol of the ETAPE database:

```bash
$ export TUNE_DIR=${TRAIN_DIR}/tune/Etape.SpeakerDiarization.TV.development
$ pyannote-speech-detection apply \
          ${TUNE_DIR} \                # <tune_dir>
          Etape.SpeakerDiarization.TV  # <database.task.protocol>
```

Among other files, this will create a file `Etape.SpeakerDiarization.TV.test.mdtm` in `APPLY_DIR` (defined below) containing speech segments.

```bash
$ export APPLY_DIR=${TUNE_DIR}/apply
$ head -n 5 $APPLY_DIR/Etape.SpeakerDiarization.TV.test.mdtm
BFMTV_BFMStory_2011-05-31_175900 1 0 5.6 speaker NA _ A
BFMTV_BFMStory_2011-05-31_175900 1 8.39 4.3 speaker NA _ B
BFMTV_BFMStory_2011-05-31_175900 1 13.09 10.81 speaker NA _ C
BFMTV_BFMStory_2011-05-31_175900 1 23.96 1.66 speaker NA _ D
BFMTV_BFMStory_2011-05-31_175900 1 25.71 4.74 speaker NA _ E
```

### Evaluation
([↑up to table of contents](#table-of-contents))

We can use [`pyannote.metrics`](http://pyannote.github.io/pyannote-metrics/) to evaluate the result:

```bash
$ pyannote-metrics.py detection Etape.SpeakerDiarization.TV ${APPLY_DIR}/Etape.SpeakerDiarization.TV.test.mdtm
Detection (collar = 0 ms)                 detection error rate    accuracy    precision    recall     total    false alarm      %    miss     %
--------------------------------------  ----------------------  ----------  -----------  --------  --------  -------------  -----  ------  ----
BFMTV_BFMStory_2011-05-31_175900                          2.49       97.65        98.12     99.41   2530.64          48.14   1.90   14.94  0.59
LCP_CaVousRegarde_2011-05-12_235900                       2.48       97.62        97.71     99.87   3218.07          75.46   2.34    4.31  0.13
LCP_EntreLesLignes_2011-05-06_192800                      5.01       95.26        95.36     99.85   1560.20          75.75   4.86    2.36  0.15
LCP_EntreLesLignes_2011-05-13_192800                      4.14       96.10        96.09     99.93   1492.76          60.74   4.07    1.05  0.07
LCP_PileEtFace_2011-05-26_192800                          2.38       97.71        97.80     99.87   1532.32          34.44   2.25    2.04  0.13
LCP_TopQuestions_2011-05-18_000400                       14.93       88.04        87.33     99.50    768.46         110.95  14.44    3.82  0.50
LCP_TopQuestions_2011-05-25_213800                        4.12       96.21        96.53     99.46    685.66          24.52   3.58    3.71  0.54
TV8_LaPlaceDuVillage_2011-05-03_201300                   14.62       88.07        87.80     99.17   1225.38         168.92  13.79   10.20  0.83
TV8_LaPlaceDuVillage_2011-05-12_172800                   14.52       87.70        87.54     99.66   1285.96         182.36  14.18    4.39  0.34
TOTAL                                                     5.79       94.69        94.80     99.67  14299.47         781.26   5.46   46.83  0.33
```

That's all folks!
