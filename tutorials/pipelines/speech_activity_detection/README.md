> The MIT License (MIT)
>
> Copyright (c) 2019-2020 CNRS
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

# Speech activity detection pipeline with `pyannote.audio`

Training a model for speech activity detection is not enough to get actual speech activity detection results. One has to also tune detection thresholds (and other optional pipeline hyper-parameters).

This tutorial assumes that you have already followed the [data preparation](../../data_preparation) tutorial, and teaches how to optimize a speech activity detection pipeline using `pyannote-pipeline` command line tool.

For simplicity, we will use a pretrained speech activity detection model.

## Table of contents
- [Citation](#citation)
- [Configuration](#configuration)
- [Training](#training)
- [Validation](#validation)
- [Application](#application)
- [More options](#more-options)

## Citation
([↑up to table of contents](#table-of-contents))

If you use `pyannote-audio` for speech activity detection, please cite the following papers:

```bibtex
@inproceedings{Bredin2020,
  Title = {{pyannote.audio: neural building blocks for speaker diarization}},
  Author = {{Bredin}, Herv{\'e} and {Yin}, Ruiqing and {Coria}, Juan Manuel and {Gelly}, Gregory and {Korshunov}, Pavel and {Lavechin}, Marvin and {Fustes}, Diego and {Titeux}, Hadrien and {Bouaziz}, Wassim and {Gill}, Marie-Philippe},
  Booktitle = {ICASSP 2020, IEEE International Conference on Acoustics, Speech, and Signal Processing},
  Address = {Barcelona, Spain},
  Month = {May},
  Year = {2020},
}
```

```bibtex
@inproceedings{Lavechin2020,
  Title = {{End-to-end Domain-Adversarial Voice Activity Detection}},
  Author = {{Lavechin}, Marvin and {Gill}, Marie-Philippe and {Bousbib}, Ruben and {Bredin}, Herv{\'e} and {Garcia-Perera}, Leibny Paola},
  Booktitle = {ICASSP 2020, IEEE International Conference on Acoustics, Speech, and Signal Processing},
  Address = {Barcelona, Spain},
  Month = {May},
  Year = {2020},
}
```

## Raw scores extraction
([↑up to table of contents](#table-of-contents))

This tutorial relies on a speech activity detection model pretrained on DIHARD dataset - but you could (should?) obviously use a locally [trained](../../models/speech_activity_detection) or [fine-tuned](../../finetune) model.

We start by extracting raw scores using `sad_dihard` pretrained model:

```bash
$ export EXP_DIR=tutorials/pipelines/speech_activity_detection
$ pyannote-audio emb apply --pretrained=sad_dihard --subset=development ${EXP_DIR} AMI.SpeakerDiarization.MixHeadset
$ export RAW_DIR=${EXP_DIR}/sad_dihard
```

Note that this is a good idea to also run this command on the `test` subset if you want to later apply the trained pipeline on them.

## Configuration
([↑up to table of contents](#table-of-contents))

To ensure reproducibility, `pyannote-pipeline` relies on a configuration file defining the experimental setup:

```bash
$ cat ${EXP_DIR}/config.yml
```
```yaml
pipeline:
   name: pyannote.audio.pipeline.speech_activity_detection.SpeechActivityDetection
   params:
      # replace {{RAW_DIR}} by its actual value
      precomputed: {{RAW_DIR}}
      
freeze:
  pad_onset: 0.0
  pad_offset: 0.0
```

This configuration file tells the pipeline to use raw speech activity detection scores that we just extracted. It also freezes two hyper-parameters that we choose not to optimize.

## Training
([↑up to table of contents](#table-of-contents))

The following command will run hyper-parameter optimization on the development subset of the AMI database. One can run it multiple times in parallel to speed things up.


```bash
$ pyannote-pipeline train --subset=development --forever ${EXP_DIR} AMI.SpeakerDiarization.MixHeadset
```

Note that we use the `development` subset for optimizing the pipeline hper-parameters because the `train` subset has usually already been used for training the model itself.

This will create a bunch of files in `TRN_DIR`, including `params.yml` that contains the (so far) optimal parameters.

```bash
$ export TRN_DIR=${EXP_DIR}/train/AMI.SpeakerDiarization.MixHeadset.development
$ cat ${TRN_DIR}/params.yml
```
```yaml
loss: 0.08679055252399374
params:
  min_duration_off: 1.0406006293815449
  min_duration_on: 0.20795742458453392
  offset: 0.7950407131756939
  onset: 0.7374970307969209
  pad_offset: 0.0
  pad_onset: 0.0
```

The `loss:` value actually corresponds to the metric that is currently being optimized. For the speech activity detection pipeline, the loss is the detection error rate.

See `pyannote.audio.pipeline.speech_activity_detection.SpeechActivityDetection` docstring for details about the `params:` section.


Note that the actual content of your `params.yml` might vary because the optimisation process is not deterministic: the longer you wait, the better it gets. We only ran 10 iterations to get `loss` down to 8.7%.

There is no easy way to decide if/when the optimization has converged to the optimal setting. The `pyannote-pipeline train` command will run forever, looking for a better set of hyper-parameters. 

## Application
([↑up to table of contents](#table-of-contents))

The optimized pipeline can then be applied on the `test` subset (as long as you also extracted correspond raw scores):

```bash
$ pyannote-audio emb apply --pretrained=sad_dihard --subset=test ${EXP_DIR} AMI.SpeakerDiarization.MixHeadset
$ pyannote-pipeline apply --subset=test ${TRN_DIR} AMI.SpeakerDiarization.MixHeadset
```

This will create a bunch of files in `${TRN_DIR}/apply/latest` subdirectory, including 
* `AMI.SpeakerDiarization.MixHeadset.test.rttm` that contains the actual output of the optimized pipeline
* `AMI.SpeakerDiarization.MixHeadset.test.eval` that provides an evaluation of the result (more or less equivalent to what you would get by using `pyannote.metrics` command line tool).


## More options

For more options, see:

```bash
$ pyannote-pipeline --help
```

That's all folks!
