#!/bin/bash
./a.out > new_tests.txt
b=$(diff new_tests.txt good_tests.txt | wc -l)
if [ "$b" -gt 0 ]; then
    echo "TESTS FAILED:"
    diff new_tests.txt good_tests.txt
fi

