#!/bin/bash
set -e
rpc_tests=$(pytest --collect-only | grep 'Function.*rpc' | cut -d' ' -f 6 | tr -d '>')

for WORKERS in {1..6}; do
    mpirun -n $WORKERS -mca orte_base_help_aggregate 0 python -m pytest tests/nn/pipe_process -k "not rpc"
    for test_name in $rpc_tests; do
        mpirun -n $WORKERS -mca orte_base_help_aggregate 0 python -m pytest tests/nn/pipe_process -k $test_name
    done
done
