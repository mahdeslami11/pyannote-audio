#!/usr/bin/env bash

export PYANNOTE_DATABASE_CONFIG=$GITHUB_WORKSPACE/tests/data/database.yml
export DEBUG=Debug.SpeakerDiarization.Debug
pyannote-audio emb train --to=4 $GITHUB_WORKSPACE/tests/cli/emb $DEBUG
pyannote-audio emb validate --from=2 --to=4 --every=2 $GITHUB_WORKSPACE/tests/cli/emb/train/$DEBUG.train $DEBUG
pyannote-audio emb apply $GITHUB_WORKSPACE/tests/cli/emb/train/$DEBUG.train/validate_diarization_fscore/$DEBUG.development $DEBUG
