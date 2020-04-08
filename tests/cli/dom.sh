#!/usr/bin/env bash

export PYANNOTE_DATABASE_CONFIG=$GITHUB_WORKSPACE/tests/data/database.yml
export DEBUG=Debug.SpeakerDiarization.Debug
pyannote-audio dom train --to=4 $GITHUB_WORKSPACE/tests/cli/dom $DEBUG
pyannote-audio dom validate --subset=train --from=2 --to=4 --every=2 $GITHUB_WORKSPACE/tests/cli/dom/train/$DEBUG.train $DEBUG
pyannote-audio dom apply $GITHUB_WORKSPACE/tests/cli/dom/train/$DEBUG.train/validate/$DEBUG.train $DEBUG
