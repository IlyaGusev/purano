import argparse
from collections import defaultdict, Counter
import numpy as np
import io
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from torch.utils.tensorboard import SummaryWriter
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import pairwise_distances
from scipy.special import expit
from razdel import tokenize
from nltk.stem.snowball import SnowballStemmer

from purano.models import Document, Info
from purano.proto.info_pb2 import EntitySpan as EntitySpanPb

# TODO: time-based embeddings
# TODO: fix time by changing distances
# TODO: agency2vec
# TODO: agglomerative clustering border
# TODO: dbscan
# TODO: clustering markup
# TODO: extracting geo


class SampleMetadata:
    def __init__(self):
        self.title = None
        self.agency = None
        self.cluster = None
        self.topic = None
        self.date = None
        self.loc = set()
        self.per = set()

    @classmethod
    def get_header(self):
        return ("title", "topic", "agency", "cluster", "date")

    def __iter__(self):
        return iter([self.title, self.topic, self.agency, self.cluster, self.date])

    def __len__(self):
        return 5

    def __repr__(self):
        assert self.title and self.agency and self.cluster is not None and self.topic is not None and self.date is not None
        return "{}\t{}\t{}\t{}\n".format(self.title, self.topic, self.agency, self.cluster, self.date)


def calc_distances(vectors, metadata,
                   fix_agencies=True, agencies_penalty=5.0,
                   fix_time=True, time_penalty=10.0,
                   fix_entities=False, entities_penalty=3.0):
    distances = pairwise_distances(vectors, metric="cosine")
    if not fix_agencies and not fix_time and not fix_entities:
        return distances

    def calc_penalty(meta1, meta2):
        penalty = 1.0
        if fix_agencies and meta1.agency == meta2.agency:
            penalty *= agencies_penalty
        if fix_time:
            time_diff = abs((meta1.date - meta2.date).total_seconds())
            x = (time_diff / 3600) - 12
            penalty *= 1.0 + expit(x) * (time_penalty - 1.0)
        if fix_entities:
            if meta1.loc and meta2.loc and not (meta1.loc & meta2.loc):
                penalty *= entities_penalty
            if meta1.per and meta2.per and not (meta1.per & meta2.per):
                penalty *= entities_penalty
        return penalty

    max_distance = distances.max()
    for i in range(len(distances)):
        for j in range(len(distances)):
            if i == j:
                continue
            penalty = calc_penalty(metadata[i], metadata[j])
            distances[i, j] = min(max_distance, distances[i, j] * penalty)
    return distances


def fetch_data(annotations, field, encode_date=False, save_entities=False):
    def encode_circular_feature(current_value, max_value):
        sin = np.sin(2*np.pi*current_value/max_value)
        cos = np.cos(2*np.pi*current_value/max_value)
        return sin, cos

    vectors = []
    metadata = []
    stemmer = None
    if save_entities:
        stemmer = SnowballStemmer("russian")
    for i, annot in enumerate(annotations):
        if i % 100 == 0:
            print(i)
        vector = np.array(annot[field])
        document = annot.document

        if encode_date:
            date = document.date
            midnight = date.replace(hour=0, minute=0, second=0, microsecond=0)
            seconds_from_midnight = (date - midnight).seconds
            sin_time, cos_time = encode_circular_feature(seconds_from_midnight, 24 * 60 * 60)

            day_of_year = date.timetuple().tm_yday
            days_in_year = date.replace(month=12, day=31).timetuple().tm_yday
            sin_day, cos_day = encode_circular_feature(day_of_year, days_in_year)
            new_features = (sin_time, cos_time, sin_day, cos_day)
            vector = np.append(vector, new_features)
        vectors.append(vector)

        meta = SampleMetadata()
        meta.topic = document.topics.strip().replace("\n", "") if annot.document.topics else "None"
        meta.agency = document.agency.host.strip().replace("\n", "")
        meta.title = document.title.replace("\n", "").strip()
        meta.date = document.date

        if save_entities:
            title_spans = annot.get_info().title_dp_ner
            text_spans = annot.get_info().text_dp_ner
            loc = list()
            per = list()
            def normalize(text):
                return [stemmer.stem(token.text) for token in tokenize(text)]

            def collect_entities(spans, text):
                for span in spans:
                    span_text = normalize(text[span.begin:span.end])
                    if span.tag == EntitySpanPb.Tag.Value("LOC"):
                        loc.extend(span_text)
                    elif span.tag == EntitySpanPb.Tag.Value("PER"):
                        per.extend(span_text)
            collect_entities(title_spans, document.title)
            collect_entities(text_spans, document.text)
            def choose_best(entities, bans):
                best = set()
                entities = Counter(entities)
                max_count = entities.most_common(1)[0][1] if entities else 0
                for entity, count in entities.items():
                    if max_count == count and entity not in bans:
                        best.add(entity)
                return best
            meta.loc = choose_best(loc, {"москв", "росс"})
            meta.per = choose_best(per, set())

        metadata.append(meta)
    vectors = np.array(vectors)
    return vectors, metadata


def run_clustering(distances, clustering_type="agglomerative"):
    if clustering_type == "agglomerative":
        clustering = AgglomerativeClustering(
            affinity="precomputed",
            distance_threshold=0.04,
            n_clusters=None,
            linkage="average")
    elif clustering_type == "dbscan":
        clustering = DBSCAN(
            min_samples=2,
            metric="precomputed",
            eps=0.02)
    labels = clustering.fit_predict(distances)
    print(labels)
    return labels


def save_to_tensorboard(vectors, metadata):
    writer = SummaryWriter()
    writer.add_embedding(vectors, metadata, metadata_header=SampleMetadata.get_header())
    writer.close()


def print_clusters_info(metadata, n=5):
    clusters = defaultdict(list)
    noisy_count = 0
    for meta in metadata:
        if int(meta.cluster) == -1:
            noisy_count += 1
            continue
        clusters[meta.cluster].append(meta)
    print("Noisy samples: {}".format(noisy_count))

    clusters = list(clusters.items())
    clusters.sort(key=lambda x: len(x[1]), reverse=True)

    max_cluster_size = len(clusters[0][1])
    print("Max cluster size: {}".format(max_cluster_size))

    clusters_count = [0 for _ in range(max_cluster_size + 1)]
    for _, cluster in clusters:
        clusters_count[len(cluster)] += 1
    for cluster_size, count in enumerate(clusters_count):
        if count == 0:
            continue
        print("{} clusters with size {}".format(count, cluster_size))

    clusters = clusters[:n]
    for cluster_num, cluster in clusters:
        print("Cluster {}: ".format(cluster_num))
        for meta in cluster:
            print(meta.title)
        print()


def cluster(db_engine, nrows, field, sort_by_date, start_date, end_date, clustering_type):
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
    vectors, metadata = fetch_data(annotations, field)
    distances = calc_distances(vectors, metadata)
    labels = run_clustering(distances, clustering_type)
    for meta, label in zip(metadata, labels):
        meta.cluster = str(label)
    save_to_tensorboard(vectors, metadata)
    print_clusters_info(metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-engine", type=str, default="sqlite:///news.db")
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--field", type=str, required=True)
    parser.add_argument("--sort-by-date", default=False,  action='store_true')
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--clustering-type", type=str, choices=("agglomerative", "dbscan"), required=True)

    args = parser.parse_args()
    cluster(**vars(args))
