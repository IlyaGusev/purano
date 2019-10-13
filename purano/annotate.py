import argparse
import json
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from bert_serving.client import BertClient
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import FastTextKeyedVectors
from razdel import tokenize

from purano.models import Document, Info
from purano.proto.info_pb2 import Info as InfoPb


class Annotator:
    def __init__(self, db_session, config_path):
        with open(config_path) as r:
            self.config = json.load(r)
        self.processors = dict()
        for key, item in self.config["processors"].items():
            item_type = item.pop("type")
            if item_type == "bert_client":
                self.processors[key] = BertClient(**item)
            elif item_type == "fasttext":
                self.processors[key] = KeyedVectors.load(item["path"])
            else:
                assert False, "Unsupported processor in config"
            print("'{}' processor loaded".format(key))
        self.db_session = db_session

    def process_by_batch(self, docs, batch_size, reannotate):
        docs_batch = []
        for doc in docs:
            docs_batch.append(doc)
            if len(docs_batch) != batch_size:
                continue
            self.annotate_batch(docs_batch, reannotate)
            docs_batch = []
        if docs_batch:
            self.annotate_batch(docs_batch, reannotate)

    def annotate_batch(self, docs_batch, reannotate):
        affected_doc_ids = tuple((doc.id for doc in docs_batch))
        info_query = self.db_session.query(Info).filter(Info.document_id.in_(affected_doc_ids))
        skip_ids = {info.document_id for info in info_query.all()}
        if skip_ids:
            if reannotate:
                info_query.delete(synchronize_session=False)
                self.db_session.commit()
                print("Annotation wiil be changed for {} documents".format(len(skip_ids)))
            else:
                docs_batch = [doc for doc in docs_batch if doc.id not in skip_ids]
                print("Skipped {} documents".format(len(skip_ids)))
                if not docs_batch:
                    return

        batch = [InfoPb() for _ in range(len(docs_batch))]
        for step_name in self.config.get("steps"):
            step = self.config[step_name]
            processor = self.processors.get(step.get("processor"))
            input_field = step.get("input_field")
            output_field = step.get("output_field")
            self.process(processor, docs_batch, batch, input_field, output_field)

        batch = [Info(doc.id, info) for doc, info in zip(docs_batch, batch)]
        self.db_session.bulk_save_objects(batch)
        self.db_session.commit()
        print("Annotated and saved {} documents, first dated {}".format(len(batch), docs_batch[0].date))

    @staticmethod
    def process(processor, docs, records_to_modify, input_field, output_field):
        inputs = [getattr(doc, input_field) for doc in docs]
        if isinstance(processor, BertClient):
            outputs = processor.encode(inputs)
        elif isinstance(processor, FastTextKeyedVectors):
            inputs = [tokenize(inp) for inp in inputs]
            outputs = []
            for sample in inputs:
                embedding = np.mean(np.array([processor.wv.word_vec(token) for token in sample]), axis=0)
                outputs.append(embedding)
        else:
            assert False
        for index, info in enumerate(records_to_modify):
            getattr(info, output_field)[:] = outputs[index]


def main(config,
         batch_size,
         db_engine,
         reannotate,
         sort_by_date,
         start_date,
         end_date,
         agency_id,
         nrows):
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
    main(**vars(args))
