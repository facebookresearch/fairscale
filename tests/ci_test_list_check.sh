#!/bin/sh

# Verify that we don't miss any tests.

find tests -name \*.py -type f| grep -v __init__.py | sort | uniq > /tmp/find.out
cat tests/ci_test_list*.txt | sort | uniq > /tmp/cat.out

if ! diff /tmp/find.out /tmp/cat.out ; then
   echo "Unit test is missing from CI"
   echo "See the diff above to fix it"
   exit 1
fi
