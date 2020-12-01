# PuraNo - news annotation and clustering

[![Build Status](https://travis-ci.com/IlyaGusev/purano.svg?branch=master)](https://travis-ci.com/IlyaGusev/purano)
[![Code Climate](https://codeclimate.com/github/IlyaGusev/purano/badges/gpa.svg)](https://codeclimate.com/github/IlyaGusev/purano)


## Installation
Install Git, DVC and pip:
```
$ sudo wget https://dvc.org/deb/dvc.list -O /etc/apt/sources.list.d/dvc.list
$ sudo apt-get update
$ sudo apt-get install git dvc python3-pip
```

Clone repo and install Python requirements (Python 3.6+ recommended):
```
$ git clone https://github.com/IlyaGusev/purano
$ python3 -m pip install -r purano/requirements.txt
```

## Run pipeline
```
$ dvc pull
$ dvc repro
$ cat output/metrics.json
```

WARNING: The clustering requires more than 8GB of RAM, as it stores all N^2 pairwise distances
