#!/bin/bash

# Reload models
rm -rf models
mkdir models
mkdir models/ruwikiruscorpora_tokens_elmo_1024_2019
cd models/ruwikiruscorpora_tokens_elmo_1024_2019 && wget http://vectors.nlpl.eu/repository/11/195.zip && unzip 195.zip && rm 195.zip && cd ../../
mkdir models/tayga_none_fasttextcbow_300_10_2019
cd models/tayga_none_fasttextcbow_300_10_2019 && wget http://vectors.nlpl.eu/repository/11/187.zip && unzip 187.zip && rm 187.zip && cd ../../

# Download datasets
mkdir datasets
cd datasets && wget -i datasets.txt && cd ../
for F in *.tar.gz; do
    cd datasets && tar -xzvf "$F" && cd ../
done
