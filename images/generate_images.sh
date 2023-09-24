#!/bin/bash

for j in 5 6 7 8 9 10 11
do
    for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
    do
        sed "s/1e7/1e$j/" config/fig$i.json > config/fig$i-$j.json
        build/symmetry run config/fig$i-$j.json images/im$i-$j.pgm 2>/dev/null
    done
done
