import argparse
import numpy as np
import io
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from torch.utils.tensorboard import SummaryWriter
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

from purano.models import Document, Info


class SampleMetadata:
    def __init__(self):
        self.title = None
        self.agency = None
        self.cluster = None
        self.topic = None

    @classmethod
    def get_header(self):
        return ("title", "topic", "agency", "cluster")

    def __iter__(self):
        return iter([self.title, self.topic, self.agency, self.cluster])

    def __len__(self):
        return 4

    def __repr__(self):
        assert self.title and self.agency and self.cluster is not None and self.topic is not None
        return "{}\t{}\t{}\t{}\n".format(self.title, self.topic, self.agency, self.cluster)


def calc_distances(vectors, metadata):
    distances = pairwise_distances(vectors, metric="cosine")
    for i in range(len(distances)):
        for j in range(len(distances)):
            if i == j:
                continue
            if metadata[i].agency == metadata[j].agency:
                distances[i, j] = 2.0
    return distances


def dump_embeddings_from_info(annotations, field):
    def to_embedding(annotation):
        return np.array(getattr(annotation.get_info(), field))

    vectors = []
    metadata = []
    for i, annot in enumerate(annotations):
        if i % 100 == 0:
            print(i)
        vectors.append(to_embedding(annot))
        document = annot.document
        meta = SampleMetadata()
        meta.topic = document.topics.strip().replace("\n", "") if annot.document.topics else "None"
        meta.agency = document.agency.host.strip().replace("\n", "")
        meta.title = document.title.replace("\n", "").strip()
        metadata.append(meta)
    vectors = np.array(vectors)
    distances = calc_distances(vectors, metadata)

    clustering = AgglomerativeClustering(
        affinity="precomputed",
        distance_threshold=0.03,
        n_clusters=None,
        linkage="average")
    labels = clustering.fit_predict(distances)
    print(labels)
    for meta, label in zip(metadata, labels):
        meta.cluster = str(label)

    writer = SummaryWriter()
    writer.add_embedding(vectors, metadata, metadata_header=SampleMetadata.get_header())
    writer.close()

    out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
    out_m = io.open('meta.tsv', 'w', encoding='utf-8')
    for vec, meta in zip(vectors, metadata):
        out_m.write("\t".join(meta) + "\n")
        out_v.write("\t".join([str(x) for x in vec]) + "\n")
    out_v.close()
    out_m.close()


def dump_embeddings(db_engine, nrows, field, sort_by_date, start_date, end_date):
    engine = create_engine(db_engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    query = session.query(Info)
    query = query.join(Info.document)
    if start_date:
        query = query.filter(Document.date > start_date)
    if end_date:
        query = query.filter(Document.date < end_date)
    query = query.join(Document.agency)
    if sort_by_date:
        query = query.order_by(Document.date)
    annotations = list(query.limit(nrows)) if nrows else list(query.all())
    dump_embeddings_from_info(annotations, field)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-engine", type=str, default="sqlite:///news.db")
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--field", type=str, required=True)
    parser.add_argument("--sort-by-date", default=False,  action='store_true')
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)

    args = parser.parse_args()
    dump_embeddings(**vars(args))
