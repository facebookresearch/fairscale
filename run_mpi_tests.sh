#!/bin/bash

set -e
for WORKERS in {1..5}; do
    mpirun -n $WORKERS python -m pytest tests/nn/pipe_process
done
