# Announcement
Open [Phd/postdoc positions](https://mycore.core-cloud.net/public.php?service=files&t=2b5f5a79d24ac81c3b3c371fcd80734b) at [LIMSI](https://www.limsi.fr/en/) combining machine learning, NLP, speech processing, and computer vision. 

# pyannote-audio

Audio processing

## Installation

```bash
$ conda create --name pyannote python=3.5 anaconda
$ source activate pyannote
$ conda install -c conda-forge yaafe
$ pip install -U pip setuptools
$ pip install pyannote.audio
```

## Citation

If you use `pyannote.audio` in your research, please use the following citation (until a regular paper is published):

```bibtex
@misc{pyannote.audio,
  author = {Bredin, H.},
  title = {pyannote.audio},
  year = {2017},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/pyannote/pyannote-audio}},
}
```

## Tutorials

 * [Feature extraction](tutorials/feature-extraction)
 * [LSTM-based speech activity detection](tutorials/speech-activity-detection)
 * [LSTM-based speaker change detection](tutorials/change-detection)
 * [_TristouNet_ neural speech turn embedding](tutorials/speaker-embedding)


## Documentation

The API is unfortunately not documented yet.
