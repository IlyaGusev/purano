#!/bin/bash

# Download datasets
rm -rf raw_datasets
mkdir raw_datasets
cd raw_datasets
wget -i ../raw_datasets.txt
for F in *.tar.gz; do
    tar -xzvf "$F"
    rm -f "$F"
done
cd ../
