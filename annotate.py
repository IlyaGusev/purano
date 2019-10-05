import argparse
import json
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from bert_serving.client import BertClient

from models import Document, Info
from info_pb2 import Info as InfoPb


class Annotator:
    def __init__(self, db_session, bert_client=None):
        self.db_session = db_session
        self.bert_client = bert_client

    def batch_annotate(self, docs, batch_size, reannotate):
        docs = list(docs)
        batch_number = 0
        while batch_number * batch_size < len(docs):
            start = batch_number * batch_size
            end = min((batch_number + 1) * batch_size, len(docs))
            if start == end:
                break

            docs_batch = docs[start:end]
            affected_doc_ids = tuple((doc.id for doc in docs_batch))
            info_query = self.db_session.query(Info).filter(Info.document_id.in_(affected_doc_ids))
            skip_ids = {info.document_id for info in info_query.all()}
            if reannotate:
                info_query.delete(synchronize_session=False)
                self.db_session.commit()
                if len(skip_ids) != 0:
                    print("Annotation wiil be replaced for {} documents".format(len(skip_ids)))
            elif skip_ids:
                docs_batch = [doc for doc in docs_batch if doc.id not in skip_ids]
                print("Skipped {} documents".format(len(skip_ids)))
                if not docs_batch:
                    batch_number += 1
                    continue

            batch = [InfoPb() for _ in range(len(docs_batch))]
            self.process_bert(docs_batch, batch)
            print("Annotated {} documents".format(len(batch)))

            batch = [Info(doc.id, info) for doc, info in zip(docs_batch, batch)]
            assert len(batch) <= batch_size
            self.db_session.bulk_save_objects(batch)
            self.db_session.commit()
            print("Saved {} documents".format(len(batch)))

            batch_number += 1

    def process_bert(self, docs, records):
        titles = [doc.title for doc in docs]
        texts = [doc.text for doc in docs]
        bert_embeddings = self.bert_client.encode(titles + texts)
        titles_embeddings = bert_embeddings[:len(titles)]
        texts_embeddings = bert_embeddings[len(titles):]
        for index, info in enumerate(records):
            info.title_bert_embedding[:] = titles_embeddings[index]
            info.text_bert_embedding[:] = texts_embeddings[index]


def main(bert_client_ip,
         bert_client_port,
         bert_client_port_out,
         batch_size,
         db_engine,
         reannotate,
         nrows):
    bert_client = BertClient(ip=bert_client_ip, port=bert_client_port, port_out=bert_client_port_out)
    engine = create_engine(db_engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    annotator = Annotator(session, bert_client)
    query = session.query(Document)
    docs = query.limit(nrows) if nrows else query.all()
    annotator.batch_annotate(docs, reannotate=reannotate, batch_size=batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert-client-ip", type=str, default="localhost")
    parser.add_argument("--bert-client-port", type=int, default=5555)
    parser.add_argument("--bert-client-port-out", type=int, default=5556)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--db-engine", type=str, default="sqlite:///news.db")
    parser.add_argument("--reannotate", default=False, action='store_true')
    parser.add_argument("--nrows", type=int, default=None)

    args = parser.parse_args()
    main(**vars(args))
