import argparse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from purano.models import Document
from purano.annotator.annotator import Annotator

def annotate(config: str,
             batch_size: int,
             db_engine: str,
             reannotate: bool,
             sort_by_date: bool,
             start_date: str,
             end_date: str,
             agency_id: int,
             nrows: int):
    assert config.endswith(".jsonnet"), "Config should be jsonnet file"
    engine = create_engine(db_engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    annotator = Annotator(session, config)
    query = session.query(Document)
    if agency_id:
        query = query.filter(Document.agency_id == agency_id)
    if start_date:
        query = query.filter(Document.date > start_date)
    if end_date:
        query = query.filter(Document.date < end_date)
    if sort_by_date:
        query = query.order_by(Document.date)
    docs = query.limit(nrows) if nrows else query.all()
    annotator.process_by_batch(docs, reannotate=reannotate, batch_size=batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--db-engine", type=str, default="sqlite:///news.db")
    parser.add_argument("--reannotate", default=False, action='store_true')
    parser.add_argument("--sort-by-date", default=False,  action='store_true')
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--agency-id", type=int, default=None)
    parser.add_argument("--nrows", type=int, default=None)

    args = parser.parse_args()
    annotate(**vars(args))
