#!/bin/bash

# Reload models
rm -rf models
mkdir models
cd models
mkdir ruwikiruscorpora_tokens_elmo_1024_2019
cd ruwikiruscorpora_tokens_elmo_1024_2019 && wget http://vectors.nlpl.eu/repository/11/195.zip && unzip 195.zip && rm 195.zip && cd ../
mkdir tayga_none_fasttextcbow_300_10_2019
cd tayga_none_fasttextcbow_300_10_2019 && wget http://vectors.nlpl.eu/repository/11/187.zip && unzip 187.zip && rm 187.zip && cd ../
wget -O bert.tar.gz http://files.deeppavlov.ai/deeppavlov_data/bert/rubert_cased_L-12_H-768_A-12_v2.tar.gz && tar -xzvf bert.tar.gz && rm bert.tar.gz && cd ../
