import argparse
import numpy as np
import io
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from torch.utils.tensorboard import SummaryWriter

from purano.models import Document, Info


def dump_embeddings_from_info(annotations):
    def to_embedding(annotation):
        return np.array(annotation.get_info().title_rvs_elmo_embedding)
    vectors = []
    metadata = []
    for i, annot in enumerate(annotations):
        vec = to_embedding(annot)
        vectors.append(vec)
        metadata.append([annot.document.title, annot.document.topics])
    vectors = np.array(vectors)
    writer = SummaryWriter()
    writer.add_embedding(vectors, metadata, metadata_header=["title", "topic"])
    writer.close()

    out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
    out_m = io.open('meta.tsv', 'w', encoding='utf-8')
    for vec, meta in zip(vectors, metadata):
        out_m.write("\t".join(meta) + "\n")
        out_v.write("\t".join([str(x) for x in vec]) + "\n")
    out_v.close()
    out_m.close()


def dump_embeddings(db_engine, nrows):
    engine = create_engine(db_engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    query = session.query(Info)
    annotations = list(query.limit(nrows)) if nrows else list(query.all())
    dump_embeddings_from_info(annotations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-engine", type=str, default="sqlite:///news.db")
    parser.add_argument("--nrows", type=int, default=1000)

    args = parser.parse_args()
    dump_embeddings(**vars(args))
