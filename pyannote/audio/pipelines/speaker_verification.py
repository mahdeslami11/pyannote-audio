# MIT License
#
# Copyright (c) 2021 CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import warnings

from pkg_resources import iter_entry_points

try:
    from functools import cached_property
except ImportError:
    from backports.cached_property import cached_property


import numpy as np
import torch

from pyannote.audio import Inference, Model, Pipeline
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines.utils import PipelineModel, get_devices, get_model


class PyannoteAudioPretrainedSpeakerEmbedding:
    """Pretrained pyannote.audio speaker embedding

    Parameters
    ----------
    embedding : PipelineModel
        pyannote.audio model
    device : torch.device, optional
        Device

    Usage
    -----
    >>> get_embedding = PyannoteAudioPretrainedSpeakerEmbedding("pyannote/embedding")
    >>> assert waveforms.ndim == 3
    >>> batch_size, num_channels, num_samples = waveforms.shape
    >>> assert num_channels == 1
    >>> embeddings = get_embedding(waveforms)
    >>> assert embeddings.ndim == 2
    >>> assert embeddings.shape[0] == batch_size

    >>> assert masks.ndim == 1
    >>> assert masks.shape[0] == batch_size
    >>> embeddings = get_embedding(waveforms, masks=masks)
    """

    def __init__(
        self,
        embedding: PipelineModel = "pyannote/embedding",
        device: torch.device = None,
    ):
        super().__init__()
        self.embedding = embedding
        self.device = device

        self.model_: Model = get_model(self.embedding)
        self.model_.eval()
        self.model_.to(self.device)

    @cached_property
    def sample_rate(self) -> int:
        return self.model_.audio.sample_rate

    @cached_property
    def dimension(self) -> int:
        return self.model_.introspection.dimension

    @cached_property
    def metric(self) -> str:
        return "cosine"

    @cached_property
    def min_num_samples(self) -> int:
        return self.model_.introspection.min_num_samples

    def __call__(
        self, waveforms: torch.Tensor, masks: torch.Tensor = None
    ) -> np.ndarray:
        with torch.no_grad():
            if masks is None:
                embeddings = self.model_(waveforms.to(self.device))
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    embeddings = self.model_(
                        waveforms.to(self.device), weights=masks.to(self.device)
                    )
        return embeddings.cpu().numpy()


def PretrainedSpeakerEmbedding(embedding: PipelineModel, device: torch.device = None):
    """Pretrained speaker embedding

    Parameters
    ----------
    embedding : Text
        Can be a SpeechBrain (e.g. "speechbrain/spkrec-ecapa-voxceleb")
        or a pyannote.audio model.
    device : torch.device, optional
        Device

    Usage
    -----
    >>> get_embedding = PretrainedSpeakerEmbedding("pyannote/embedding")
    >>> get_embedding = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb")
    >>> get_embedding = PretrainedSpeakerEmbedding("nvidia/speakerverification_en_titanet_large")
    >>> assert waveforms.ndim == 3
    >>> batch_size, num_channels, num_samples = waveforms.shape
    >>> assert num_channels == 1
    >>> embeddings = get_embedding(waveforms)
    >>> assert embeddings.ndim == 2
    >>> assert embeddings.shape[0] == batch_size

    >>> assert masks.ndim == 1
    >>> assert masks.shape[0] == batch_size
    >>> embeddings = get_embedding(waveforms, masks=masks)
    """

    Klasses = [
        o.load()
        for o in iter_entry_points(group="pyannote.audio.speaker_embedding.pretrained")
    ]

    for ExternalPretrainedSpeakerEmbedding in filter(
        lambda Klass: Klass.is_supported(embedding), Klasses
    ):
        try:
            return ExternalPretrainedSpeakerEmbedding(embedding, device=device)
        # TODO: check what kind of exception are raised
        except Exception:
            pass

    try:
        return PyannoteAudioPretrainedSpeakerEmbedding(embedding, device=device)

    except Exception:
        # TODO: check what kind of exception are raised
        raise NotImplementedError(
            f"It looks like you are trying to load '{embedding}' pretrained speaker embedding that is not supported by pyannote.audio. "
            f"You might want to try installing 'pyannote_audio_contrib' that adds support for more speaker embeddings."
        )


class SpeakerEmbedding(Pipeline):
    """Speaker embedding pipeline

    This pipeline assumes that each file contains exactly one speaker
    and extracts one single embedding from the whole file.

    Parameters
    ----------
    embedding : Model, str, or dict, optional
        Pretrained embedding model. Defaults to "pyannote/embedding".
        See pyannote.audio.pipelines.utils.get_model for supported format.
    segmentation : Model, str, or dict, optional
        Pretrained segmentation (or voice activity detection) model.
        See pyannote.audio.pipelines.utils.get_model for supported format.
        Defaults to no voice activity detection.

    Usage
    -----
    >>> from pyannote.audio.pipelines import SpeakerEmbedding
    >>> pipeline = SpeakerEmbedding()
    >>> emb1 = pipeline("speaker1.wav")
    >>> emb2 = pipeline("speaker2.wav")
    >>> from scipy.spatial.distance import cdist
    >>> distance = cdist(emb1, emb2, metric="cosine")[0,0]
    """

    def __init__(
        self,
        embedding: PipelineModel = "pyannote/embedding",
        segmentation: PipelineModel = None,
    ):
        super().__init__()

        self.embedding = embedding
        self.segmentation = segmentation

        self.embedding_model_: Model = get_model(embedding)

        if self.segmentation is None:
            models = [self.embedding_model_]
        else:
            segmentation_model: Model = get_model(self.segmentation)
            models = [self.embedding_model_, segmentation_model]

        # send models to GPU (when GPUs are available and model is not already on GPU)
        cpu_models = [model for model in models if model.device.type == "cpu"]
        for cpu_model, gpu_device in zip(
            cpu_models, get_devices(needs=len(cpu_models))
        ):
            cpu_model.to(gpu_device)

        if self.segmentation is not None:
            self.voice_activity_ = Inference(
                segmentation_model,
                pre_aggregation_hook=lambda scores: np.max(
                    scores, axis=-1, keepdims=True
                ),
            )

    def apply(self, file: AudioFile) -> np.ndarray:

        device = self.embedding_model_.device

        # read audio file and send it to GPU
        waveform = self.embedding_model_.audio(file)[0][None].to(device)

        if self.segmentation is None:
            weights = None
        else:
            # obtain voice activity scores
            weights = self.voice_activity_(file).data
            # HACK -- this should be fixed upstream
            weights[np.isnan(weights)] = 0.0
            weights = torch.from_numpy(weights**3)[None, :, 0].to(device)

        # extract speaker embedding on parts of
        with torch.no_grad():
            return self.embedding_model_(waveform, weights=weights).cpu().numpy()


def main(
    protocol: str = "VoxCeleb.SpeakerVerification.VoxCeleb1",
    subset: str = "test",
    embedding: str = "pyannote/embedding",
    segmentation: str = None,
):

    import typer
    from pyannote.database import FileFinder, get_protocol
    from pyannote.metrics.binary_classification import det_curve
    from scipy.spatial.distance import cdist
    from tqdm import tqdm

    pipeline = SpeakerEmbedding(embedding=embedding, segmentation=segmentation)

    protocol = get_protocol(protocol, preprocessors={"audio": FileFinder()})

    y_true, y_pred = [], []

    emb = dict()

    trials = getattr(protocol, f"{subset}_trial")()

    for t, trial in enumerate(tqdm(trials)):

        audio1 = trial["file1"]["audio"]
        if audio1 not in emb:
            emb[audio1] = pipeline(audio1)

        audio2 = trial["file2"]["audio"]
        if audio2 not in emb:
            emb[audio2] = pipeline(audio2)

        y_pred.append(cdist(emb[audio1], emb[audio2], metric="cosine")[0][0])
        y_true.append(trial["reference"])

    _, _, _, eer = det_curve(y_true, np.array(y_pred), distances=True)
    typer.echo(
        f"{protocol.name} | {subset} | {embedding} | {segmentation} | EER = {100 * eer:.3f}%"
    )


if __name__ == "__main__":
    import typer

    typer.run(main)
