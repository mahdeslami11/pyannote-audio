import pytest
import pytorch_lightning as pl
import torch
from torch.nn import Module
from torch_audiomentations import Compose, Gain

from pyannote.audio.augmentation.registry import (
    register_augmentation,
    unregister_augmentation,
)
from pyannote.audio.models.segmentation.debug import SimpleSegmentationModel
from pyannote.audio.tasks import VoiceActivityDetection
from pyannote.database import FileFinder, get_protocol


class RandomAugmentation(Module):
    "Fake Noise tranform"

    def forward(self, waveforms):
        if not self.training:
            return waveforms
        return torch.ones_like(waveforms)


def test_can_register_augmentation():
    for when in ["output", "input"]:
        net = Module()
        aug = RandomAugmentation()
        register_augmentation(aug, net, when=when)
        assert hasattr(net, "__augmentation")
        assert net.__augmentation[when] == aug


def test_can_unregistrer_augmentation():
    net = Module()
    register_augmentation(RandomAugmentation(), net, when="output")
    unregister_augmentation(net, when="output")
    assert not hasattr(net.__augmentation, "output")


def test_fail_unregister_augmentation():
    with pytest.raises(ValueError):
        unregister_augmentation(RandomAugmentation(), when="output")


def test_compose_augmentation_sample_rate():
    protocol = get_protocol(
        "Debug.SpeakerDiarization.Debug", preprocessors={"audio": FileFinder()}
    )
    sr = 8000
    tfm = Gain(sample_rate=sr)
    vad = VoiceActivityDetection(
        protocol,
        duration=2.0,
        batch_size=32,
        num_workers=4,
        augmentation=Compose([tfm]),
    )
    model = SimpleSegmentationModel(task=vad)
    trainer = pl.Trainer(fast_dev_run=True)
    _ = trainer.fit(model)
    assert tfm.sample_rate == model.hparams.sample_rate
