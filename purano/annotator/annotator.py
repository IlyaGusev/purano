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
from purano.annotator.ner_client import NerClient
from purano.annotator.bert_processor import BertProcessor


class Annotator:
    def __init__(self, db_session, config_path: str):
        self.config = json.loads(_jsonnet.evaluate_file(config_path))
        self.processors = dict()
        for key, item in self.config["processors"].items():
            item_type = item.pop("type")
            if item_type == "bert_client":
                self.processors[key] = BertClient(**item)
            elif item_type == "bert":
                self.processors[key] = BertProcessor(**item)
            elif item_type == "fasttext":
                self.processors[key] = KeyedVectors.load(item["path"])
            elif item_type == "elmo":
                self.processors[key] = ElmoEmbedder(**item)
            elif item_type == "ner_client":
                self.processors[key] = NerClient(**item)
            else:
                assert False, "Unsupported processor in config"
            print("'{}' processor loaded".format(key))
        self.db_session = db_session

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
        batch = [InfoPb() for _ in range(len(docs_batch))]
        step_time = {}
        for step_name in self.config.get("steps"):
            step = copy.deepcopy(self.config[step_name])
            processor_name = step.pop("processor")
            processor = self.processors.get(processor_name)
            input_fields = step.pop("input_fields")
            output_field = step.pop("output_field")
            outputs = self.process(processor, docs_batch, input_fields, **step)
            step_time[step_name] = time.time() - start_time
            start_time = time.time()
            for index, info in enumerate(batch):
                getattr(info, output_field).extend(outputs[index])

        start_time = time.time()
        batch = [Info(doc.id, info) for doc, info in zip(docs_batch, batch)]
        self.db_session.bulk_save_objects(batch)
        self.db_session.commit()
        save_time = time.time() - start_time

        print("Annotated and saved {} documents, last dated {}".format(len(batch), docs_batch[-1].date))
        print("Processing time:")
        for p, t in step_time.items():
            print("{}: {:.2f} seconds".format(p, t))
        print("Saving time: {:.2f} seconds".format(save_time))

    @staticmethod
    def process(processor, docs: List[Document], input_fields: List[str], **kwargs):
        assert processor is not None
        assert len(docs) > 0

        inputs = ["\n".join([getattr(doc, field) for field in input_fields]) for doc in docs]
        if isinstance(processor, (BertClient, BertProcessor, NerClient)):
            return processor.encode(inputs)

        agg_type = kwargs.pop("agg_type")
        assert agg_type in ("mean", "max", "mean||max")
        max_tokens_count = kwargs.pop("max_tokens_count", 100)
        samples_tokens = [[token.text for token in tokenize(inp)][:max_tokens_count] for inp in inputs]
        batch_size = len(samples_tokens)
        max_tokens_count = max([len(sample) for sample in samples_tokens])
        if isinstance(processor, FastTextKeyedVectors):
            embeddings = np.zeros((batch_size, max_tokens_count, processor.wv.vector_size), dtype=np.float64)
            for batch_num, sample in enumerate(samples_tokens):
                for token_num, token in enumerate(sample):
                    embeddings[batch_num, token_num, :] = processor.wv.word_vec(token)
        elif isinstance(processor, ElmoEmbedder):
            embeddings = processor.batch_to_embeddings(samples_tokens)[0].cpu().numpy()
            embeddings = embeddings.swapaxes(1, 2)
            embeddings = embeddings.reshape(embeddings.shape[0], embeddings.shape[1], -1)
        else:
            assert False
        mean_embeddings = np.mean(embeddings, axis=1)
        max_embeddings = np.max(embeddings, axis=1)
        if agg_type == "mean":
            return mean_embeddings
        if agg_type == "max":
            return max_embeddings
        if agg_type == "mean||max":
            return np.concatenate((mean_embeddings, max_embeddings), axis=1)
        else:
            assert False

