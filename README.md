# Announcement
Open [Phd/postdoc positions](https://mycore.core-cloud.net/public.php?service=files&t=2b5f5a79d24ac81c3b3c371fcd80734b) at [LIMSI](https://www.limsi.fr/en/) combining machine learning, NLP, speech processing, and computer vision.

# pyannote-audio

Neural building blocks for speaker diarization: 
* speech activity detection
* speaker change detection
* speaker embedding
* speaker diarization pipeline


## Installation

```bash
$ conda create --name pyannote python=3.6 anaconda
$ source activate pyannote
$ pip install pyannote.audio
```

## Citation

If you use `pyannote.audio` please use the following citations.

  - Speech  activity and speaker change detection
    ```bibtex
    @inproceedings{Yin2017,
      Author = {Ruiqing Yin and Herv\'e Bredin and Claude Barras},
      Title = {{Speaker Change Detection in Broadcast TV using Bidirectional Long Short-Term Memory Networks}},
      Booktitle = {{18th Annual Conference of the International Speech Communication Association, Interspeech 2017}},
      Year = {2017},
      Month = {August},
      Address = {Stockholm, Sweden},
      Url = {https://github.com/yinruiqing/change_detection}
    }
    ```
  - Speaker embedding
    ```bibtex
    @inproceedings{Bredin2017,
        author = {Herv\'{e} Bredin},
        title = {{TristouNet: Triplet Loss for Speaker Turn Embedding}},
        booktitle = {42nd IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2017},
        year = {2017},
        url = {http://arxiv.org/abs/1609.04301},
    }
    ```
  - Speaker diarization pipeline
    ```bibtex
    @inproceedings{Yin2018,
      Author = {Ruiqing Yin and Herv\'e Bredin and Claude Barras},
      Title = {{Neural Speech Turn Segmentation and Affinity Propagation for Speaker Diarization}},
      Booktitle = {{19th Annual Conference of the International Speech Communication Association, Interspeech 2018}},
      Year = {2018},
      Month = {September},
      Address = {Hyderabad, India},
    }
    ```

## Tutorials

  * [Feature extraction](tutorials/feature-extraction)
  * Models
    * [LSTM-based speech activity detection](tutorials/models/speech-activity-detection)
    * [LSTM-based speaker change detection](tutorials/models/speaker-change-detection)
    * [LSTM-based speaker embedding](tutorials/models/speaker-embedding)
  * Pipelines
    * [Speech activity detection pipeline](tutorials/pipelines/speech-activity-detection)

## Documentation

The API is unfortunately not documented yet.
