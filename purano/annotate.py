import argparse
import json
import time
import copy
import numpy as np
import torch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from bert_serving.client import BertClient
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import FastTextKeyedVectors
from allennlp.commands.elmo import ElmoEmbedder
from razdel import tokenize
from transformers import BertTokenizer, BertModel

from purano.models import Document, Info
from purano.proto.info_pb2 import Info as InfoPb


class BertProcessor:
    def __init__(self, pretrained_model_name_or_path):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = BertModel.from_pretrained(pretrained_model_name_or_path, output_hidden_states=True)
        self.max_seq_len = 64

    def encode(self, docs):
        batch_input_ids = torch.zeros((len(docs), self.max_seq_len), dtype=int)
        batch_mask = torch.zeros((len(docs), self.max_seq_len), dtype=int)
        for i, sample in enumerate(docs):
            input_ids = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + self.tokenizer.tokenize(sample)[:self.max_seq_len-2] + ['[SEP]'])
            pad_len = self.max_seq_len - len(input_ids)
            input_mask = [1] * len(input_ids) + [0] * pad_len
            input_ids += [0] * pad_len
            batch_input_ids[i, :] = torch.tensor(input_ids)
            batch_mask[i, :] = torch.tensor(input_mask)

        self.model.eval()
        with torch.no_grad():
            all_hidden_states = self.model(batch_input_ids, attention_mask=batch_mask)[-1]
        embeddings = all_hidden_states[-2].cpu().numpy()
        embeddings = embeddings.mean(axis=1)
        return embeddings


class Annotator:
    def __init__(self, db_session, config_path):
        with open(config_path) as r:
            self.config = json.load(r)
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

        start_time = time.time()
        batch = [InfoPb() for _ in range(len(docs_batch))]
        for step_name in self.config.get("steps"):
            step = copy.deepcopy(self.config[step_name])
            processor = self.processors.get(step.pop("processor"))
            input_field = step.pop("input_field")
            output_field = step.pop("output_field")
            outputs = self.process(processor, docs_batch, input_field, **step)
            for index, info in enumerate(batch):
                getattr(info, output_field)[:] = outputs[index]

        process_time = time.time() - start_time

        start_time = time.time()
        batch = [Info(doc.id, info) for doc, info in zip(docs_batch, batch)]
        self.db_session.bulk_save_objects(batch)
        self.db_session.commit()
        save_time = time.time() - start_time

        print("Annotated and saved {} documents, first dated {}".format(len(batch), docs_batch[0].date))
        print("Processing time: {:.2f} seconds, saving time: {:.2f} seconds".format(process_time, save_time))

    @staticmethod
    def process(processor, docs, input_field, **kwargs):
        assert processor is not None
        assert len(docs) > 0

        inputs = [getattr(doc, input_field) for doc in docs]
        if isinstance(processor, BertClient) or isinstance(processor, BertProcessor):
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


def annotate(config,
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
    annotate(**vars(args))
