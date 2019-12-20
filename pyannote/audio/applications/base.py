#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2017-2019 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr

import io
import time
import yaml
import zipfile
import hashlib
import torch
import multiprocessing
from typing_extensions import Literal
from typing import Optional, Union
from pathlib import Path
from os.path import basename
import numpy as np
from tqdm import tqdm
from glob import glob
from pyannote.database import FileFinder
from pyannote.database import get_protocol
from pyannote.database import get_annotated
from pyannote.audio.features.utils import get_audio_duration
from sortedcontainers import SortedDict
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from pyannote.core.utils.helper import get_class_by_name
import warnings
from pyannote.audio.train.task import Task

from pyannote.audio.features import Precomputed

def create_zip(validate_dir: Path):
    """

    # create zip file containing:
    # config.yml
    # {self.train_dir_}/specs.yml
    # {self.train_dir_}/weights/{epoch:04d}*.pt
    # {self.validate_dir_}/params.yml

    """

    existing_zips = list(validate_dir.glob('*.zip'))
    if len(existing_zips) == 1:
        existing_zips[0].unlink()
    elif len(existing_zips) > 1:
        msg = (
            f'Looks like there are too many torch.hub zip files '
            f'in {validate_dir}.')
        raise NotImplementedError(msg)

    params_yml = validate_dir / 'params.yml'

    with open(params_yml, 'r') as fp:
        params = yaml.load(fp, Loader=yaml.SafeLoader)
        epoch = params['epoch']

    xp_dir = validate_dir.parents[3]
    config_yml = xp_dir / 'config.yml'

    train_dir = validate_dir.parents[1]
    weights_dir = train_dir / 'weights'
    specs_yml = train_dir / 'specs.yml'

    hub_zip = validate_dir / 'hub.zip'
    with zipfile.ZipFile(hub_zip, 'w') as z:
        z.write(config_yml, arcname=config_yml.relative_to(xp_dir))
        z.write(specs_yml, arcname=specs_yml.relative_to(xp_dir))
        z.write(params_yml, arcname=params_yml.relative_to(xp_dir))
        for pt in weights_dir.glob(f'{epoch:04d}*.pt'):
            z.write(pt, arcname=pt.relative_to(xp_dir))

    sha256_hash = hashlib.sha256()
    with open(hub_zip,"rb") as fp:
        for byte_block in iter(lambda: fp.read(4096),b""):
            sha256_hash.update(byte_block)

    hash_prefix = sha256_hash.hexdigest()[:10]
    target = validate_dir / f"{hash_prefix}.zip"
    hub_zip.rename(target)

    return target


class Application:

    CONFIG_YML = '{experiment_dir}/config.yml'
    TRAIN_DIR = '{experiment_dir}/train/{protocol}.{subset}'
    WEIGHTS_DIR = '{train_dir}/weights'
    MODEL_PT = '{train_dir}/weights/{epoch:04d}.pt'
    VALIDATE_DIR = '{train_dir}/validate{_criterion}/{protocol}.{subset}'
    APPLY_DIR = '{validate_dir}/apply/{epoch:04d}'

    @classmethod
    def from_train_dir(cls, train_dir: Path,
                            training: bool = False):

        app = cls(train_dir.parents[1], training=training)
        app.train_dir_ = train_dir
        return app

    @classmethod
    def from_model_pt(cls, model_pt: Path,
                           training: bool = False):

        train_dir = model_pt.parents[1]
        app = cls.from_train_dir(model_pt.parents[1], training=training)

        app.model_pt_ = model_pt
        epoch = int(model_pt.stem)
        app.model_ = app.load_model(epoch)
        app.epoch_ = epoch
        return app

    @classmethod
    def from_validate_dir(cls, validate_dir: Path,
                               training: Optional[bool] = False):

        # load params.yml file from validate directory
        with open(validate_dir / 'params.yml', 'r') as fp:
            params_yml = yaml.load(fp, Loader=yaml.SafeLoader)

        # build path to best epoch model
        epoch = params_yml['epoch']
        model_pt = Path(cls.MODEL_PT.format(
            train_dir=validate_dir.parents[1], epoch=epoch))

        # instantiate application
        # TODO. get rid of from_model_pt
        app = cls.from_model_pt(model_pt, training=training)
        app.validate_dir_ = validate_dir
        app.epoch_ = epoch

        # keep track of pipeline parameters
        app.pipeline_params_ = params_yml.get('params', {})

        return app

    def __init__(self, experiment_dir, training=False):
        """

        Parameters
        ----------
        experiment_dir : str
        training : boolean, optional
            When False, data augmentation is disabled.
        """

        self.experiment_dir = experiment_dir

        # load configuration
        config_yml = self.CONFIG_YML.format(experiment_dir=self.experiment_dir)
        with open(config_yml, 'r') as fp:
            self.config_ = yaml.load(fp, Loader=yaml.SafeLoader)

        # preprocessors
        preprocessors = dict()

        for key, preprocessor in self.config_.get('preprocessors', {}).items():
            # preprocessors:
            #    key:
            #       name: package.module.ClassName
            #       params:
            #          param1: value1
            #          param2: value2
            if isinstance(preprocessor, dict):
                Klass = get_class_by_name(preprocessor['name'])
                preprocessors[key] = Klass(**preprocessor.get('params', {}))
                continue

            try:
                # preprocessors:
                #    key: /path/to/database.yml
                preprocessors[key] = FileFinder(preprocessor)

            except FileNotFoundError as e:
                # preprocessors:
                #    key: /path/to/{uri}.wav
                preprocessors[key] = preprocessor

        if 'audio' not in preprocessors:
            preprocessors['audio'] = FileFinder()

        if 'duration' not in preprocessors:
            preprocessors['duration'] = get_audio_duration

        self.preprocessors_ = preprocessors

        # scheduler
        SCHEDULER_DEFAULT = {'name': 'DavisKingScheduler',
                             'params': {'learning_rate': 'auto'}}
        scheduler_cfg = self.config_.get('scheduler', SCHEDULER_DEFAULT)
        Scheduler = get_class_by_name(
            scheduler_cfg['name'],
            default_module_name='pyannote.audio.train.schedulers')
        scheduler_params = scheduler_cfg.get('params', {})
        self.learning_rate_ = scheduler_params.pop('learning_rate', 'auto')
        self.scheduler_ = Scheduler(**scheduler_params)

        # optimizer
        OPTIMIZER_DEFAULT = {
            'name': 'SGD',
            'params': {'momentum': 0.9, 'dampening': 0, 'weight_decay': 0,
                       'nesterov': True}}
        optimizer_cfg = self.config_.get('optimizer', OPTIMIZER_DEFAULT)
        try:
            Optimizer = get_class_by_name(optimizer_cfg['name'],
                                          default_module_name='torch.optim')
            optimizer_params = optimizer_cfg.get('params', {})
            self.get_optimizer_ = partial(Optimizer, **optimizer_params)

        # do not raise an error here as it is possible that the optimizer is
        # not really needed (e.g. in pipeline training)
        except ModuleNotFoundError as e:
            warnings.warn(e.args[0])

        # data augmentation (only when training the model)
        if training and 'data_augmentation' in self.config_ :
            DataAugmentation = get_class_by_name(
                self.config_['data_augmentation']['name'],
                default_module_name='pyannote.audio.augmentation')
            augmentation = DataAugmentation(
                **self.config_['data_augmentation'].get('params', {}))
        else:
            augmentation = None

        # custom callbacks
        self.callbacks_ = []
        for callback_config in self.config_.get('callbacks', {}):
            Callback = get_class_by_name(callback_config['name'])
            callback = Callback(**callback_config.get('params', {}))
            self.callbacks_.append(callback)

        # feature extraction
        FEATURE_DEFAULT = {'name': 'RawAudio',
                           'params': {'sample_rate': 16000}}
        feature_cfg = self.config_.get('feature_extraction', FEATURE_DEFAULT)
        FeatureExtraction = get_class_by_name(
            feature_cfg['name'],
            default_module_name='pyannote.audio.features')
        feature_params = feature_cfg.get('params', {})
        self.feature_extraction_ = FeatureExtraction(
            **feature_params,
            augmentation=augmentation)

        # task
        TaskClass = get_class_by_name(
            self.config_['task']['name'],
            default_module_name=self.config_default_module)
        self.task_ = TaskClass(
            **self.config_['task'].get('params', {}))

        # architecture
        Architecture = get_class_by_name(
            self.config_['architecture']['name'],
            default_module_name='pyannote.audio.models')
        params = self.config_['architecture'].get('params', {})

        self.get_model_from_specs_ = partial(Architecture, **params)
        self.model_resolution_ = Architecture.get_resolution(**params)
        self.model_alignment_ =  Architecture.get_alignment(**params)


    def train(self, protocol_name: str,
                    subset: str = 'train',
                    warm_start: Union[int, Literal['last'], Path] = 0,
                    epochs: int = 1000,
                    device: Optional[torch.device] = None,
                    n_jobs: int = 1):
        """Train model

        Parameters
        ----------
        protocol_name : `str`
        subset : {'train', 'development', 'test'}, optional
            Defaults to 'train'.
        warm_start : `int`, "last", or `Path`, optional
            When `int`, restart training at this epoch.
            When "last", restart from last epoch.
            When `Path`, restart from this model checkpoint.
            Defaults to training from scratch (warm_start = 0).
        epochs : `int`, optional
            Train for that many epochs. Defaults to 1000.
        device : `torch.device`, optional
            Device on which the model will be allocated. Defaults to using CPU.
        n_jobs : `int`, optional
        """

        # initialize batch generator
        protocol = get_protocol(protocol_name, progress=True,
                                preprocessors=self.preprocessors_)

        batch_generator = self.task_.get_batch_generator(
            self.feature_extraction_,
            protocol,
            subset=subset,
            resolution=self.model_resolution_,
            alignment=self.model_alignment_)

        # initialize model architecture based on specifications
        model = self.get_model_from_specs_(batch_generator.specifications)

        train_dir = Path(self.TRAIN_DIR.format(
            experiment_dir=self.experiment_dir,
            protocol=protocol_name,
            subset=subset))

        # use last available epoch as starting point
        if warm_start == 'last':
            warm_start = self.get_number_of_epochs(train_dir=train_dir) - 1

        iterations = self.task_.fit_iter(
            model,
            batch_generator,
            warm_start=warm_start,
            epochs=epochs,
            get_optimizer=self.get_optimizer_,
            scheduler=self.scheduler_,
            learning_rate=self.learning_rate_,
            train_dir=train_dir,
            device=device,
            callbacks=self.callbacks_,
            n_jobs=n_jobs)

        for _ in iterations:
            pass

    def load_model(self,
                   epoch: int,
                   train_dir: Optional[Path] = None):
        """Load pretrained model

        Parameters
        ----------
        epoch : int
            Which epoch to load.
        train_dir : str, optional
            Path to train directory. Defaults to self.train_dir_.
        """

        if train_dir is None:
            train_dir = self.train_dir_

        # initialize model from specs stored on disk
        specs_yml = self.task_.SPECS_YML.format(train_dir=train_dir)
        with io.open(specs_yml, 'r') as fp:
            specifications = yaml.load(fp, Loader=yaml.SafeLoader)
        specifications['task'] = Task.from_str(specifications['task'])
        self.model_ = self.get_model_from_specs_(specifications)

        import torch
        weights_pt = self.MODEL_PT.format(
            train_dir=train_dir, epoch=epoch)

        # if GPU is not available, load using CPU
        self.model_.load_state_dict(
            torch.load(weights_pt, map_location=lambda storage, loc: storage))

        return self.model_

    def get_number_of_epochs(self, train_dir=None, return_first=False):
        """Get information about completed epochs

        Parameters
        ----------
        train_dir : str, optional
            Training directory. Defaults to self.train_dir_
        return_first : bool, optional
            Defaults (False) to return number of epochs.
            Set to True to also return index of first epoch.

        """

        if train_dir is None:
            train_dir = self.train_dir_

        directory = self.MODEL_PT.format(train_dir=train_dir, epoch=0)[:-7]
        weights = sorted(glob(directory + '*[0-9][0-9][0-9][0-9].pt'))

        if not weights:
            number_of_epochs = 0
            first_epoch = None

        else:
            number_of_epochs = int(basename(weights[-1])[:-3]) + 1
            first_epoch = int(basename(weights[0])[:-3])

        return (number_of_epochs, first_epoch) if return_first \
                                               else number_of_epochs

    def validate_init(self, protocol_name,
                            subset='development'):
        raise NotImplementedError('')

    def validate_epoch(self, epoch,
                             validation_data,
                             protocol=None,
                             subset='development',
                             device: Optional[torch.device] = None,
                             batch_size: int = 32,
                             n_jobs: int = 1,
                             **kwargs):

        raise NotImplementedError('')

    def validation_criterion(self, **kwargs):
        return None

    def validate(self, protocol: str,
                       subset: str = 'development',
                       every: int = 1,
                       start: Union[int, Literal['last']] = 1,
                       end: Union[int, Literal['last']] = 100,
                       chronological: bool = False,
                       device: Optional[torch.device] = None,
                       batch_size: int = 32,
                       n_jobs: int = 1,
                       **kwargs):

        # use last available epoch as starting point
        if start == 'last':
            start = self.get_number_of_epochs() - 1

        # use last available epoch as end point
        if end == 'last':
            end = self.get_number_of_epochs() - 1

        criterion = self.validation_criterion(**kwargs)

        validate_dir = Path(self.VALIDATE_DIR.format(
            train_dir=self.train_dir_,
            _criterion=f'_{criterion}' if criterion is not None else '',
            protocol=protocol, subset=subset))

        params_yml = validate_dir / 'params.yml'

        validate_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(validate_dir),
                               purge_step=start)

        validation_data = self.validate_init(protocol, subset=subset)

        if n_jobs > 1:
            self.pool_ = multiprocessing.Pool(n_jobs)

        progress_bar = tqdm(unit='iteration')

        for i, epoch in enumerate(
            self.validate_iter(start=start, end=end, step=every,
                               chronological=chronological)):

            # {'metric': 'detection_error_rate',
            #  'minimize': True,
            #  'value': 0.9,
            #  'pipeline': ...}
            details = self.validate_epoch(epoch,
                                          validation_data,
                                          protocol=protocol,
                                          subset=subset,
                                          device=device,
                                          batch_size=batch_size,
                                          n_jobs=n_jobs,
                                          **kwargs)

            # initialize
            if i == 0:
                # what is the name of the metric?
                metric = details['metric']
                # should the metric be minimized?
                minimize = details['minimize']
                # epoch -> value dictionary
                values = SortedDict()

            # metric value for current epoch
            values[epoch] = details['value']

            # send value to tensorboard
            writer.add_scalar(
                f'validate/{protocol}.{subset}/{metric}',
                values[epoch], global_step=epoch)

            # keep track of best value so far
            if minimize:
                best_epoch = values.iloc[np.argmin(values.values())]
                best_value = values[best_epoch]

            else:
                best_epoch = values.iloc[np.argmax(values.values())]
                best_value = values[best_epoch]

            # if current epoch leads to the best metric so far
            # store both epoch number and best pipeline parameter to disk
            if best_epoch == epoch:

                best = {
                    metric: best_value,
                    'epoch': epoch,
                }
                if 'pipeline' in details:
                    pipeline = details['pipeline']
                    best['params'] = pipeline.parameters(instantiated=True)
                with open(params_yml, mode='w') as fp:
                    fp.write(yaml.dump(best, default_flow_style=False))

                # create/update zip file for later upload to torch.hub
                hub_zip = create_zip(validate_dir)

            # progress bar
            desc = (f'{metric} | '
                    f'Epoch #{best_epoch} = {100 * best_value:g}% (best) | '
                    f'Epoch #{epoch} = {100 * details["value"]:g}%')
            progress_bar.set_description(desc=desc)
            progress_bar.update(1)

    def validate_iter(self, start=1, end=None, step=1, sleep=10,
                      chronological=False):
        """Continuously watches `train_dir` for newly completed epochs
        and yields them for validation

        Note that epochs will not necessarily be yielded in order.
        The very last completed epoch will always be first on the list.

        Parameters
        ----------
        start : int, optional
            Start validating after `start` epochs. Defaults to 1.
        end : int, optional
            Stop validating after epoch `end`. Defaults to never stop.
        step : int, optional
            Validate every `step`th epoch. Defaults to 1.
        sleep : int, optional
        chronological : bool, optional
            Force chronological validation.

        Usage
        -----
        >>> for epoch in app.validate_iter():
        ...     app.validate(epoch)


        """

        if end is None:
            end = np.inf

        validated_epochs = set()
        next_epoch_to_validate_in_order = start

        while next_epoch_to_validate_in_order < end:

            # wait for first epoch to complete
            _, first_epoch = self.get_number_of_epochs(return_first=True)
            if first_epoch is None:
                print('waiting for first epoch to complete...')
                time.sleep(sleep)
                continue

            # corner case: make sure this does not wait forever
            # for epoch 'start' as it might never happen, in case
            # training is started after n pre-existing epochs
            if next_epoch_to_validate_in_order < first_epoch:
                next_epoch_to_validate_in_order = first_epoch

            # first epoch has completed
            break

        while True:

            # check last completed epoch
            last_completed_epoch = self.get_number_of_epochs() - 1

            # if last completed epoch has not been processed yet,
            # always process it first (except if 'in order')
            if (not chronological) and (last_completed_epoch not in validated_epochs):
                next_epoch_to_validate = last_completed_epoch
                time.sleep(5)  # HACK give checkpoint time to save weights

            # in case no new epoch has completed since last time
            # process the next epoch in chronological order (if available)
            elif next_epoch_to_validate_in_order <= last_completed_epoch:
                next_epoch_to_validate = next_epoch_to_validate_in_order

            # otherwise, just wait for a new epoch to complete
            else:
                time.sleep(sleep)
                continue

            if next_epoch_to_validate not in validated_epochs:

                # yield next epoch to process
                yield next_epoch_to_validate

                # stop validation when the last epoch has been reached
                if next_epoch_to_validate >= end:
                    return

                # remember which epoch was processed
                validated_epochs.add(next_epoch_to_validate)

            # increment 'chronological' processing
            if next_epoch_to_validate_in_order == next_epoch_to_validate:
                next_epoch_to_validate_in_order += step


    def apply(self, protocol_name: str,
                    subset: Optional[str] = "test",
                    duration: Optional[float] = None,
                    step: float = 0.25,
                    device: Optional[torch.device] = None,
                    batch_size: int = 32,
                    **kwargs):
        """Apply pre-trained model

        Parameters
        ----------
        protocol_name : `str`
        step : `float`, optional

        subset : {'train', 'development', 'test'}
            Defaults to 'test'
        """

        model = self.model_.to(device)
        model.eval()

        output_dir = Path(self.APPLY_DIR.format(
            validate_dir=self.validate_dir_,
            epoch=self.epoch_))

        # do not use memmap as this would lead to too many open files
        if isinstance(self.feature_extraction_, Precomputed):
            self.feature_extraction_.use_memmap = False

        # initialize extraction
        extraction = self.Extraction(
            model=model,
            feature_extraction=self.feature_extraction_,
            duration=duration,
            step=step * duration,
            batch_size=batch_size,
            device=device)

        params = {}
        try:
            params['labels'] = model.classes
        except AttributeError as e:
            pass
        try:
            params['dimension'] = model.dimension
        except AttributeError as e:
            pass

        # create metadata file at root that contains
        # sliding window and dimension information
        precomputed = Precomputed(
            root_dir=output_dir,
            sliding_window=extraction.sliding_window,
            **params)

        # file generator
        protocol = get_protocol(protocol_name, progress=True,
                                preprocessors=self.preprocessors_)

        for current_file in getattr(protocol, subset)():
            fX = extraction(current_file)
            precomputed.dump(current_file, fX)

        # do not proceed with the full pipeline
        # when there is no such thing for current task
        if not hasattr(self, 'Pipeline'):
            return

        # instantiate pipeline
        pipeline = self.Pipeline(scores=output_dir)
        pipeline.instantiate(self.pipeline_params_)

        # load pipeline metric (when available)
        try:
            metric = pipeline.get_metric()
        except NotImplementedError as e:
            metric = None

        # apply pipeline and dump output to RTTM files
        output_rttm = output_dir / f'{protocol_name}.{subset}.rttm'
        with open(output_rttm, 'w') as fp:
            for current_file in getattr(protocol, subset)():
                hypothesis = pipeline(current_file)
                pipeline.write_rttm(fp, hypothesis)

                # compute evaluation metric (when possible)
                if 'annotation' not in current_file:
                    metric = None

                # compute evaluation metric (when available)
                if metric is None:
                    continue

                reference = current_file['annotation']
                uem = get_annotated(current_file)
                _ = metric(reference, hypothesis, uem=uem)

        # print pipeline metric (when available)
        if metric is None:
            return

        output_eval = output_dir / f'{protocol_name}.{subset}.eval'
        with open(output_eval, 'w') as fp:
            fp.write(str(metric))
