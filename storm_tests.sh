#!/bin/bash
FILES="random_scram/*"
for f in $FILES
do
	/usr/bin/time --format "\t%e" --output=logs/storm.csv -a ~/Desktop/storm/build/bin/storm-dft -dft $f --timebound 1 --bdd --modularisation --timeout 60
	/usr/bin/time --format "\t%e" --output=logs/storm_full.csv -a ~/Desktop/storm/build/bin/storm-dft -dft $f --timebound-all 1 --bdd --modularisation --timeout 60
done
