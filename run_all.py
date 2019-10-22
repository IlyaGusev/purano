import os
import argparse

from purano.parse_csv import parse_csv
from purano.annotate import annotate
from purano.train_clf import train_clf

def run_all(db_engine):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-engine", type=str, default="sqlite:///news.db")
    args = parser.parse_args()
    run_all(**vars(args))
