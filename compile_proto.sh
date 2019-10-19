# !/bin/bash

cd purano/proto
protoc *.proto --python_out=.
cd ../../
