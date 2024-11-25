#!/bin/bash

cd data/data_split

echo "Downloading Proposed Split Version 2.0."
wget http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip
unzip xlsa17.zip
rm -rf xlsa17.zip