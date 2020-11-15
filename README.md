# PuraNo - news annotation and clustering

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
