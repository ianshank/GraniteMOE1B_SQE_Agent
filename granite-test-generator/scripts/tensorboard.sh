#!/usr/bin/env bash
set -euo pipefail

tensorboard --logdir "${TB_LOG_DIR:-runs}" --port "${TB_PORT:-6006}" --host 0.0.0.0
