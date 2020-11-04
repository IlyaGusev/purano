import json
import time
import copy
from typing import List, Iterable

import _jsonnet
import numpy as np
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import FastTextKeyedVectors
from allennlp.commands.elmo import ElmoEmbedder
from razdel import tokenize
from bert_serving.client import BertClient

from purano.models import Document, Info
from purano.proto.info_pb2 import Info as InfoPb, EntitySpan as EntitySpanPb
from purano.annotator.processors import Processor


class Annotator:
    def __init__(self, db_session, config_path: str):
        self.db_session = db_session
        self.config = json.loads(_jsonnet.evaluate_file(config_path))
        self.processors = dict()
        for key, item in self.config["processors"].items():
            item_type = item.pop("type")
            self.processors[key] = Processor.by_name(item_type)(**item)
            print("'{}' processor loaded".format(key))

    def process_by_batch(self, docs: Iterable[Document], batch_size: int, reannotate: bool):
        docs_batch = []
        for doc in docs:
            docs_batch.append(doc)
            if len(docs_batch) != batch_size:
                continue
            self.annotate_batch(docs_batch, reannotate)
            docs_batch = []
        if docs_batch:
            self.annotate_batch(docs_batch, reannotate)

    def annotate_batch(self, docs_batch: List[Document], reannotate: bool):
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

        start_time = time.time()
        infos_batch = [InfoPb() for _ in range(len(docs_batch))]
        step_time = {}
        for step_name in self.config.get("steps"):
            step = copy.deepcopy(self.config[step_name])
            processor_name = step.pop("processor")
            processor = self.processors.get(processor_name)
            assert processor is not None, "No processor with name '{}'".format(processor_name)
            processor(docs_batch, infos_batch, **step)
            step_time[step_name] = time.time() - start_time
            start_time = time.time()

        start_time = time.time()
        batch = [Info(doc.id, info) for doc, info in zip(docs_batch, infos_batch)]
        self.db_session.bulk_save_objects(batch)
        self.db_session.commit()
        save_time = time.time() - start_time

        print("Annotated and saved {} documents, last dated {}".format(len(batch), docs_batch[-1].date))
        print("Processing time:")
        for p, t in step_time.items():
            print("{}: {:.2f} seconds".format(p, t))
        print("Saving time: {:.2f} seconds".format(save_time))
