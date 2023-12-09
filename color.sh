#!/bin/bash

echo "config/blue.csv" | entr bash -c "echo 'Colorizing $2 with $1' && build/symmetry color $1 $2"

