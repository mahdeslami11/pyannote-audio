# pyannote-audio

Audio processing

## Installation

```bash
$ conda create --name pyannote python=3.5 anaconda
$ source activate pyannote
$ conda install -c conda-forge yaafe
$ pip install pyannote.audio
```

What did you just install?

- [`keras`](keras.io) (and its [`theano`](http://deeplearning.net/software/theano/) backend) is used for all things deep.
- [`yaafe`](https://github.com/Yaafe/Yaafe) is used for MFCC feature extraction.
  You might also want to checkout [`librosa`](http://librosa.github.io) (easy to install, but much slower) though `pyannote.audio` does not support it yet (pull requests are welcome, though!)
- [`pyannote.audio`](http://pyannote.github.io) is this library.

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

 * [LSTM-based speech activity detection](tutorials/speech-activity-detection)

* Feature extraction

    ```bash
    $ pyannote-speech-feature.py --help
    ```
    
 * LSTM-based speaker change detection:

    ```bash
    $ pyannote-change-detection.py --help
    ```

## Documentation

The API is unfortunately not documented yet.
