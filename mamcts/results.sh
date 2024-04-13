#!/bin/bash

for ((i=1; i<=100; i++)); do
    filename="test_${i}.log.txt"
    if [ -f "$filename" ]; then
        echo "Contents of $filename:"
        cat "$filename"
    	echo "----------------------------------------"
    fi
done

