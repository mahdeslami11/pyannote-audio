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

# Neural speech turn embedding with `pyannote.audio`

In this tutorial, you will learn how to train a [_TristouNet_](http://arxiv.org/abs/1609.04301) speech turn embedding using `pyannote-speaker-embedding` command line tool.

## Table of contents
- [Citation](#citation)
- [Installation](#installation)
- [Experimental setup](#experimental-setup)
  - [ETAPE database](#etape-database)
  - [Data preparation](#data-preparation)
  - [Training](#training)
  - [Validation](#validation)
  - [Extraction](#extraction)
- [Pretrained model](#pretrained-model)

## Citation
([↑up to table of contents](#table-of-contents))

If you use `pyannote-audio` for speaker (or audio) neural embedding, please cite the following paper:

```bibtex
@inproceedings{Bredin2017,
    author = {Herv\'{e} Bredin},
    title = {{TristouNet: Triplet Loss for Speaker Turn Embedding}},
    booktitle = {42nd IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2017},
    year = {2017},
    url = {http://arxiv.org/abs/1609.04301},
}
```

## Installation
([↑up to table of contents](#table-of-contents))

```bash
$ conda create --name py35-pyannote-audio python=3.5 anaconda
$ source activate py35-pyannote-audio
$ conda install -c conda-forge yaafe
$ pip install "pyannote.audio==0.3"
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

### Data preparation
([↑up to table of contents](#table-of-contents))

`pyannote-speaker-embedding` relies on a configuration file defining the feature extraction step:

```bash
$ export ROOT_DIR=tutorials/speaker-embedding
$ cat $ROOT_DIR/config.yml
feature_extraction:
   name: YaafeMFCC
   params:
      duration: 0.032
      step: 0.020
      coefs: 11
      D: True
      DD: True
      e: False
      De: True
      DDe: True
```

The following command should provide you with some details on those parameters:

```python
$ ipython
>>> from pyannote.audio.features import YaafeMFCC
>>> help(YaafeMFCC)
```

Running the following command will then prepare the data later used for training.

```bash
$ pyannote-speaker-embedding data --duration=2 --step=0.5 $ROOT_DIR Etape.SpeakerDiarization.TV
Training set: 28it [03:11,  3.26s/it]
Development set: 9it [00:51,  4.33s/it]
Test set: 9it [00:50,  4.61s/it]
```

Practically, this will generate a large collection of speaker-labeled 2s segments (with a 500ms step) stored in HDF5 file format:

```bash
$ ls $ROOT_DIR/2+0.5/sequences
ls $ROOT_DIR/2+0.5/sequences
Etape.SpeakerDiarization.TV.development.h5  Etape.SpeakerDiarization.TV.test.h5  Etape.SpeakerDiarization.TV.train.h5
```

### Training
([↑up to table of contents](#table-of-contents))

`pyannote-speaker-embedding` then relies on a second configuration file defining the neural network architecture and the training setup  (loss, batch size, number of sequences per speaker, etc...):

```
$ export EXPERIMENT_DIR=$ROOT_DIR/2+0.5/TristouNet
$ cat $EXPERIMENT_DIR/config.yml
architecture:
   name: TristouNet
   params:
     rnn: LSTM
     recurrent: [16]
     mlp: [16, 16]
     bidirectional: concat

approach:
   name: TripletLoss
   params:
     metric: euclidean
     margin: 0.1
     clamp: positive
     per_batch: 12
     per_fold: 20
     per_label: 3
     gradient_factor: 10000
     batch_size: 32
```

The following commands should provide you with some details on those parameters:

```python
$ ipython
>>> from pyannote.audio.embedding.models import TristouNet
>>> help(TristouNet)
>>> from pyannote.audio.embedding.approach import TripletLoss
>>> help(TripletLoss)
```

Running the following command will actually train the network:

```bash
$ pyannote-speaker-embedding train --subset=train $EXPERIMENT_DIR Etape.SpeakerDiarization.TV
Epoch 1/1000
1/1 [==============================] - 439s - loss: 0.0781
Epoch 2/1000
1/1 [==============================] - 33s - loss: 0.0345
...
Epoch 998/1000
1/1 [==============================] - 32s - loss: 0.0056
Epoch 999/1000
1/1 [==============================] - 32s - loss: 0.0072
Epoch 1000/1000
1/1 [==============================] - 30s - loss: 0.0071
```

Practically, this will generate a bunch of files in the `TRAIN_DIR` directory:

```bash
$ export TRAIN_DIR=$EXPERIMENT_DIR/train/Etape.SpeakerDiarization.TV.train
$ ls $TRAIN_DIR
loss.train.eps
loss.train.png
loss.train.txt
weights
```

`$TRAIN_DIR/loss.train.txt` contains the value of the (hopefully decreasing) loss after each epoch:

```bash
$ cat $TRAIN_DIR/loss.train.txt
0 2017-07-05T14:16:29.608407 0.078
1 2017-07-05T14:17:03.749268 0.034
2 2017-07-05T14:17:33.785134 0.023
...
997 2017-07-05T23:07:05.195969 0.006
998 2017-07-05T23:07:37.625752 0.007
999 2017-07-05T23:08:08.165979 0.007
```

![$TRAIN_DIR/loss.train.png](2+0.5/TristouNet/train/Etape.SpeakerDiarization.TV.train/loss.train.png)

The trained neural network is dumped into the `$TRAIN_DIR/weights` subdirectory after each epoch:

```bash
$ ls $TRAIN_DIR/weights
0000.h5  0001.h5  0002.h5  [...]  0997.h5  0998.h5  0999.h5
```

### Validation
([↑up to table of contents](#table-of-contents))

Now that the network is training, we can run the following command in parallel to perform validation (here, on the development set).
It will continuously watch the `$TRAIN_DIR/weights` repository for completed epochs, perform a "same/different" validation experiment after each epoch, and report the equal error rate (EER):

```
$ pyannote-speaker-embedding validate --subset=development $TRAIN_DIR Etape.SpeakerDiarization.TV
Best EER = 11.49% @ epoch #986 :: EER = 12.41% @ epoch #388 :: 389epoch [8:52:25, 27.47s/epoch]
```

Practically, this will generate a bunch of files in the `VALIDATION_DIR` directory:

```
$ export VALIDATION_DIR=$TRAIN_DIR/validate/Etape.SpeakerDiarization.TV
$ ls $VALIDATION_DIR
development.eer.eps
development.eer.png
development.eer.txt
```

`$VALIDATION_DIR/development.eer.txt` contains the value of equal error rate after each epoch:

```bash
$ cat $VALIDATION_DIR/development.eer.txt
0000 0.262921
0001 0.253302
0002 0.257871
...
0997 0.115512
0998 0.117362
0999 0.115682
```

File `$VALIDATION_DIR/development.eer.txt` can then be used to select the best epoch.

![$VALIDATION_DIR/development.eer.png](2+0.5/TristouNet/train/Etape.SpeakerDiarization.TV.train/validate/Etape.SpeakerDiarization.TV/development.eer.png)

### Extraction
([↑up to table of contents](#table-of-contents))

Once the model is trained and validated, one can use the following command to extract embeddings using the best epoch:

```bash
$ export $OUTPUT_DIR=/path/where/to/export/embeddings
$ pyannote-speaker-embedding apply --step=2.0 $VALIDATION_DIR/development.eer.txt Etape.SpeakerDiarization.TV $OUTPUT_DIR
Development set: 9it [00:30,  2.90s/it]
Test set: 9it [00:31,  3.07s/it]
Training set: 28it [01:47,  2.27s/it]
```

These embeddings can then be used in the following way:

```python
$ ipython
>>> # get first test file
>>> from pyannote.database import get_protocol
>>> protocol = get_protocol('Etape.SpeakerDiarization.TV')
>>> first_test_file = next(protocol.test())

>>> # load previously computed embedding
>>> from pyannote.audio.features import Precomputed
>>> precomputed = Precomputed('/path/where/to/export/embeddings')
>>> embedding = precomputed(first_test_file)

>>> # print embedding of first five positions of the 2s sliding window
>>> for i, (X, window) in enumerate(embedding.iterfeatures(window=True)):
...     print(window)
...     print(X)
...     print()
...     if i > 4: break
...
[ 00:00:00.000 -->  00:00:02.000]
[ 0.03328229 -0.43043402 -0.09652223  0.34882328  0.00291051 -0.24498068
 -0.1679723   0.38746014  0.14270751  0.02056508  0.12392229  0.23357467
 -0.09522723 -0.05860756  0.56213856  0.15828927]

[ 00:00:02.000 -->  00:00:04.000]
[-0.00286854 -0.13661338 -0.20620684  0.19647488 -0.14323252 -0.22506329
 -0.2959002   0.48400506  0.38754445 -0.1947242  -0.05701421  0.26069036
  0.06632474  0.04767529  0.41845512  0.25714415]

[ 00:00:04.000 -->  00:00:06.000]
[ 0.18890585 -0.26475263 -0.19176343  0.22359782 -0.04932787 -0.27070776
 -0.22572571  0.51305044  0.21585359 -0.18941438  0.13768911  0.05140146
  0.20741057  0.08183527  0.46214265  0.22405304]

[ 00:00:06.000 -->  00:00:08.000]
[ 0.2594575  -0.18474458 -0.22103274  0.04060838 -0.06001015 -0.14333209
 -0.36431456  0.57409704  0.24390684 -0.2181658  -0.02526117 -0.20132062
  0.39483786  0.13541959  0.03222574  0.19508976]

[ 00:00:08.000 -->  00:00:10.000]
[ 0.30861437 -0.25210094 -0.21497072 -0.1467725  -0.11706238  0.31691527
 -0.06859872  0.11231507  0.11751619  0.24540658  0.12666012 -0.61729604
 -0.04183983 -0.27335858  0.11984541 -0.28287509]

[ 00:00:10.000 -->  00:00:12.000]
[ 0.22853695  0.18382129 -0.31430557 -0.40747666 -0.13695037 -0.04599557
 -0.31592551  0.45766941  0.25655341  0.03845279 -0.13544272 -0.29190832
  0.34690803 -0.14094749  0.06348757  0.06259846]
```

## Pretrained model
([↑up to table of contents](#table-of-contents))

A _TristouNet_ model (trained and validated on ETAPE database) is available for you to test directly in `tutorials/speaker-embedding/2+0.5/TristouNet`.

## Going further...
([↑up to table of contents](#table-of-contents))

```
$ pyannote-speaker-embedding --help
```
