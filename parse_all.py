import argparse
import os
from tempfile import NamedTemporaryFile
import pandas as pd
from sqlalchemy import create_engine

from models import Base, Document


def fix_line_feed(input_file_name, output_file_name):
    with open(input_file_name, "r") as r, open(output_file_name, "w") as w:
        for line in r:
            line = line.replace("\\n", "\\\\n")
            w.write(line)


def process_parser_data(file_name):
    dataset = pd.read_csv(
        file_name, sep=',', quotechar='\"', escapechar='\\',
        encoding='utf-8', error_bad_lines=False, header=0,
        verbose=False, keep_date_col=True, index_col=False)
    dataset = dataset[["date", "url", "edition", "title", "text", "authors", "topics"]]
    dataset["date"] = pd.to_datetime(dataset["date"])
    dataset["text"] = dataset["text"].apply(lambda x: x.replace("\\n", "\n"))
    dataset["edition"] = dataset["edition"].apply(lambda x: None if x == "-" else x)
    print(dataset.info())
    return dataset


def main(db_engine, files):
    engine = create_engine(db_engine)
    Base.metadata.create_all(engine, Base.metadata.tables.values(),checkfirst=True)

    for file_name in files:
        temp_file = NamedTemporaryFile(delete=False)
        fix_line_feed(file_name, temp_file.name)
        dataset = process_parser_data(temp_file.name)
        temp_file.close()
        os.unlink(temp_file.name)
        print(dataset.iloc[0]["text"])
        #dataset.to_sql(Document.__tablename__, engine.raw_connection(), if_exists='append', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--files',
        nargs='+',
        type=str,
        dest='files',
        required=True
    )
    parser.add_argument("--db-engine", type=str, default="sqlite:///news.db")

    args = parser.parse_args()
    main(**vars(args))
