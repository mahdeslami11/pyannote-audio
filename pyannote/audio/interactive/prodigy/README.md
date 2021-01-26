


## Installation

Setup a `conda` environment with Python 3.8.
Install [`pytorch`](https://pytorch.org)
Install `develop` branch of [`pyannote.audio`](https://github.com/pyannote/pyannote-audio/tree/develop)
```bash
pip install https://github.com/pyannote/pyannote-audio/archive/develop.zip
```

Install [`faiss`](https://github.com/facebookresearch/faiss)
```bash
 conda install -c pytorch faiss-cpu
```

Install [`prodigy`](https://prodi.gy/)


## Usage

```bash
prodigy pyannote.speaker my_dataset speech_turns.jsonl -speakers=speakers.txt -F speaker.py
```

where `speech_turns.jsonl` contains one line per speech turn to annotate:

```
{"audio": "/path/to/audio1.wav", "start": 0.5, "end": 5.4}
{"audio": "/path/to/audio1.wav", "start": 6.4, "end": 10.2}
{"audio": "/path/to/audio1.wav", "start": 10.2, "end": 37.}
{"audio": "/path/to/audio2.wav", "start": 2.3, "end": 8.56}
{"audio": "/path/to/audio2.wav", "start": 9.0, "end": 15.4}
{"audio": "/path/to/audio2.wav", "start": 17.4, "end": 21.2}
```

and `speakers.txt` (optional) contains the list of speakers:

```
sheldon_cooper
penny
```

Use `-allow-new-speakers` to allow the creation of speakers that are not in the list.
Without this option, only speakers in `speakers.txt` (when provided) are allowed.

```bash
prodigy pyannote.speaker my_dataset speech_turns.jsonl -speakers=speakers.txt -allow-new-speaker -F speaker.py
```
