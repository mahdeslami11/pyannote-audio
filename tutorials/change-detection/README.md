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
> Ruiqing Yin
> Hervé Bredin - http://herve.niderb.fr

# Speaker change detection with `pyannote.audio`

In this tutorial, you will learn how to train, tune, and and test a speaker change detector based on MFCCs and LSTMs, using `pyannote-change-detection` command line tool.

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

If you use `pyannote-audio` for (speaker) change detection, please cite the following paper:

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

To ensure reproducibility, `pyannote-change-detection` relies on a configuration file defining the experimental setup:

```bash
$ cat tutorials/change-detection/config.yml
feature_extraction:
   name: YaafeMFCC
   params:
      e: False                   # this experiments relies
      De: True                   # on 11 MFCC coefficients
      DDe: True                  # with 1st and 2nd derivatives
      D: True                    # without energy, but with
      DD: True                   # energy derivatives
      stack: 1

architecture:
   name: StackedLSTM
   params:                       # this experiments relies
     n_classes: 1                # on one LSTM layer (16 outputs)
     lstm: [16]                  # and one dense layer.
     mlp: [16]                   # LSTM is bidirectional
     bidirectional: 'concat'     # LSTM is bidirectional
     final_activation: 'sigmoid'

sequences:
   duration: 3.2                 # this experiments relies on sliding windows
   step: 0.8                     # of 3.2s with a step of 0.8s
   balance: 0.05                 # and balancing neighborhood size of 0.05s

```

### Training
([↑up to table of contents](#table-of-contents))

The following command will train the network using the training set of the `TV` protocol of the ETAPE database. This may take a long time...

```bash
$ export EXPERIMENT_DIR=tutorials/change-detection
$ pyannote-change-detection train \       #  
          ${EXPERIMENT_DIR} \             # <experiment_dir>
          Etape.SpeakerDiarization.TV     # <database.task.protocol>
Epoch 1/100
62464/62464 [==============================] - 171s - loss: 0.1543 - acc: 0.9669   
Epoch 2/100
62464/62464 [==============================] - 117s - loss: 0.1375 - acc: 0.9692     
Epoch 3/100
62464/62464 [==============================] - 115s - loss: 0.1376 - acc: 0.9691     
...
Epoch 50/100
62464/62464 [==============================] - 112s - loss: 0.0903 - acc: 0.9724  
...

```

This will create a bunch of files in `TRAIN_DIR` (defined below), including plots showing the accuracy epoch after epoch.

In the rest of this tutorial, we assume that we killed training after epoch #50.

### Validation
([↑up to table of contents](#table-of-contents))

To get a quick idea of how the network is doing during training, one can use the "validate" mode.
It can (should!) be run in parallel to training and evaluates the model epoch after epoch.

```bash
$ export TRAIN_DIR=${EXPERIMENT_DIR}/train/Etape.SpeakerDiarization.TV.train
$ pyannote-change-detection validate \
          ${TRAIN_DIR} \               # <train_dir>
          Etape.SpeakerDiarization.TV  # <database.task.protocol>
```

This will create a bunch of files in `TRAIN_DIR/validate`, including plots showing the evolution of coverage (at a given purity)  epoch after epoch.

### Tuning
([↑up to table of contents](#table-of-contents))

Now that the network is trained, we need to tune a bunch of hyper-parameters (including which epoch to use, [peak detection thresholds and minimum segment duration](https://github.com/pyannote/pyannote-audio/blob/e3d683b65a6ace52d9de1f5d027ec71d7e9a08e2/pyannote/audio/signal.py#L51-L54)...

This is done on the developement set of the `TV` protocol of the ETAPE database. This may also take a long time...

```bash
$ export TRAIN_DIR=${EXPERIMENT_DIR}/train/Etape.SpeakerDiarization.TV.train
$ pyannote-change-detection tune \
          ${TRAIN_DIR} \               # <train_dir>
          Etape.SpeakerDiarization.TV  # <database.task.protocol>
```


This will create a `tune.yml` file in `TUNE_DIR` (defined below) containing the best set of hyper-parameters:

```yaml
$ cat ${TUNE_DIR}/tune.yml
status:
  epochs: 22                        # 22 epochs were available for tuning
  objective: 0.9412983557344607     # best (1 - coverage)
alpha: 0.27812651949435974          # best peak detection threshold
epoch: 8                            # best epoch is #8
min_duration: 2.8372675488582666    # best minimum duration
```

### Testing
([↑up to table of contents](#table-of-contents))

Now that the change detector is trained and tuned, we can apply it on the test set of the `TV` protocol of the ETAPE database:

```bash
$ export TUNE_DIR=${TRAIN_DIR}/tune/Etape.SpeakerDiarization.TV.development
$ pyannote-change-detection apply \
          ${TUNE_DIR} \                # <tune_dir>
          Etape.SpeakerDiarization.TV  # <database.task.protocol>
```

Among other files, this will create a file `Etape.SpeakerDiarization.TV.test.mdtm` in `APPLY_DIR` (defined below) containing (hopefully homonegenous) segments.

```bash
$ export APPLY_DIR=${TUNE_DIR}/apply
$ head -n 5 $APPLY_DIR/Etape.SpeakerDiarization.TV.test.mdtm
BFMTV_BFMStory_2011-05-31_175900 1 -0.0125 4.5425 speaker NA _ A
BFMTV_BFMStory_2011-05-31_175900 1 4.53 8.66 speaker NA _ B
BFMTV_BFMStory_2011-05-31_175900 1 13.19 4.71 speaker NA _ C
BFMTV_BFMStory_2011-05-31_175900 1 17.9 3.67 speaker NA _ D
BFMTV_BFMStory_2011-05-31_175900 1 21.57 4.5 speaker NA _ E
```

### Evaluation
([↑up to table of contents](#table-of-contents))

We can use [`pyannote.metrics`](http://pyannote.github.io/pyannote-metrics/) to evaluate the result:

```bash
$ pyannote-metrics.py segmentation Etape.SpeakerDiarization.TV ${APPLY_DIR}/Etape.SpeakerDiarization.TV.test.mdtm
Segmentation (tolerance = 500 ms)      coverage    purity    precision    recall
-----------------------------------  ----------  --------  -----------  --------
BFMTV_BFMStory_2011-05-31_175900          53.91     96.07         5.52     12.90
LCP_CaVousRegarde_2011-05-12_235900       59.54     84.41        11.83     11.96
...
TOTAL                                     .....     .....         ....     .....
```

That's all folks!




## Going further...

```bash
$ pyannote-change-detection --help
```
