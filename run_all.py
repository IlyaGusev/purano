import os
import argparse

from purano.parse_csv import parse_csv
from purano.annotate import annotate
from purano.train_clf import train_clf

DATASETS_PATH = "datasets"


def run_all(db_engine, config, batch_size):
    parse_csv(db_engine, [file_path for file_path in os.listdir(DATASETS_PATH) if file_path.endswith(".csv")])
    annotate(config, batch_size, db_engine, reannotate=False,
             sort_by_date=True, start_date="2019-01-01", end_date="2019-03-01",
             agency_id=None, nrows=None):



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-engine", type=str, default="sqlite:///news.db")
    parser.add_argument("--config", type=str, default="annotator.json")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    run_all(**vars(args))
