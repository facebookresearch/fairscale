#!/bin/sh

# Verify that we don't miss any tests.

find_total=`find tests -name \*.py -type f| grep -v __init__.py | sort | uniq | wc -l`
grep_total=`grep py tests/ci_test_list*.txt | sort | uniq | wc -l`

echo $find_total
echo $grep_total

if [ $find_total != $grep_total ]; then
   echo "unit test is missing from CI"
   exit 1
fi
