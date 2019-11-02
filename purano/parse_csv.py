import argparse
import os
from urllib.parse import urlsplit
from tempfile import NamedTemporaryFile
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from purano.models import Base, Document, Agency


def fix_line_feed(input_file_name, output_file_name):
    with open(input_file_name, "r") as r, open(output_file_name, "w") as w:
        for line in r:
            line = line.replace("\\n", "\\\\n").replace("\\r", " ").replace("\t", " ")
            w.write(line)


def process_parser_data(file_name):
    dataset = pd.read_csv(
        file_name, sep=',', quotechar='\"', escapechar='\\',
        encoding='utf-8', error_bad_lines=False, header=0,
        verbose=False, keep_date_col=True, index_col=False)
    dataset = dataset[["date", "url", "edition", "title", "text", "authors", "topics"]]
    dataset = dataset[(~dataset["text"].isnull() & ~dataset["title"].isnull())]
    dataset["date"] = pd.to_datetime(dataset["date"])
    dataset["text"] = dataset["text"].apply(lambda x: x.replace("\\n", " "))
    dataset["edition"] = dataset["edition"].apply(lambda x: None if x == "-" else x)
    # TODO: sort by date
    # TODO: undup
    print(dataset.info())
    print(dataset.head(5))
    return dataset


def parse_csv(db_engine, files):
    engine = create_engine(db_engine)
    Base.metadata.create_all(engine, Base.metadata.tables.values(),checkfirst=True)

    for file_name in files:
        temp_file = NamedTemporaryFile(delete=False)
        fix_line_feed(file_name, temp_file.name)
        dataset = process_parser_data(temp_file.name)
        temp_file.close()
        os.unlink(temp_file.name)

        Session = sessionmaker(bind=engine)
        session = Session()
        host = dataset["url"].apply(lambda x: urlsplit(x).netloc).value_counts().index[0]
        agency = session.query(Agency).filter_by(host=host).first()
        if agency is None:
            agency = Agency()
            agency.host = host
            session.add(agency)
            session.commit()
        print("Agency id: {}".format(agency.id))
        dataset["agency_id"] = agency.id
        print(dataset.iloc[0]["text"])
        dataset.to_sql(Document.__tablename__, engine.raw_connection(), if_exists='append', index=False)


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
    parse_csv(**vars(args))
