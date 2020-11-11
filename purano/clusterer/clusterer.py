import json
import itertools
from typing import Optional
from collections import defaultdict, Counter

import numpy as np
from _jsonnet import evaluate_file as jsonnet_evaluate_file
from nltk.stem.snowball import SnowballStemmer
from razdel import tokenize
from scipy.special import expit
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import pairwise_distances
from torch.utils.tensorboard import SummaryWriter

from purano.models import Document
from purano.proto.info_pb2 import EntitySpan as EntitySpanPb

# TODO: agency2vec
# TODO: clustering markup
# TODO: extracting geo


class Clusterer:
    def __init__(self, db_session, config_path: str):
        self.clusterings = {
            "agglomerative": AgglomerativeClustering,
            "dbscan": DBSCAN
        }
        self.stemmer = SnowballStemmer("russian")
        self.config = json.loads(jsonnet_evaluate_file(config_path))
        self.db_session = db_session
        self.vectors = None

        self.num2doc = list()
        self.num2entities = list()

        self.id2num = dict()
        self.host2nums = defaultdict(list)

        self.labels = dict()
        self.clusters = defaultdict(list)

    def fetch_embeddings(self,
        start_date: Optional[str]=None,
        end_date: Optional[str]=None,
        sort_by_date: Optional[bool]=False,
        nrows: Optional[int]=None
    ):
        embedding_key = self.config.pop("embedding_key")
        entities_key = self.config.pop("entities_key", None)

        query = self.db_session.query(Document)
        query = query.join(Document.info)
        if start_date:
            query = query.filter(Document.date > start_date)
        if end_date:
            query = query.filter(Document.date < end_date)
        if sort_by_date:
            query = query.order_by(Document.date)
        documents = list(query.limit(nrows)) if nrows else list(query.all())

        vectors = []
        for doc_num, document in enumerate(documents):
            if doc_num % 100 == 0:
                print("Fetched {} documents".format(doc_num))
            vector = np.array(document.info[embedding_key])

            entities_pb = document.info[entities_key]
            entities = self.collect_entities(entities_pb, document.title) if entities_key else dict()

            encoded_date = self.encode_date(document.date)

            self.db_session.expunge(document)
            del document.info

            vectors.append(vector)
            self.num2doc.append(document)
            self.num2entities.append(entities)
            self.id2num[document.id] = doc_num
            self.host2nums[document.host].append(doc_num)
        self.vectors = np.array(vectors)

    def cluster(self):
        distances = pairwise_distances(self.vectors, metric="cosine")
        fix_hosts = self.config.pop("fix_hosts", False)
        fix_time = self.config.pop("fix_time", False)
        hosts_penalty = self.config.pop("hosts_penalty", 1.0)
        time_penalty = self.config.pop("time_penalty", 1.0)
        max_distance = distances.max()

        if fix_hosts:
            penalty = hosts_penalty
            for host, nums in self.host2nums.items():
                for i, j in itertools.product(nums, repeat=2):
                    if i == j:
                        continue
                    distances[i, j] = min(max_distance, distances[i, j] * penalty)

        if fix_time:
            def calc_time_penalty(doc1, doc2):
                time_diff = abs((doc1.date - doc2.date).total_seconds())
                x = (time_diff / 3600) - 12
                return 1.0 + expit(x) * (time_penalty - 1.0)

            for i in range(len(distances)):
                if i % 100 == 0:
                    print(i)
                for j in range(len(distances)):
                    if i == j:
                        continue
                    penalty = calc_penalty(self.num2doc[i], self.num2doc[j])
                    distances[i, j] = min(max_distance, distances[i, j] * penalty)

        clustering_type = self.config.pop("clustering_type")
        clustering_params = self.config.pop("clustering_params")
        clustering = self.clusterings[clustering_type](**clustering_params)
        labels = clustering.fit_predict(distances)
        for label, doc in zip (labels, self.num2doc):
            self.labels[doc.id] = label
            self.clusters[label].append(doc.id)

    def print_clusters(self, n=5):
        noisy_count = len(self.clusters.get(-1)) if -1 in self.clusters else 0
        print("Noisy samples: {}".format(noisy_count))

        clusters = list(self.clusters.items())
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
            for doc_id in cluster:
                print(self.num2doc[self.id2num[doc_id]].title)
            print()

    def collect_entities(self, spans, text):
        loc = []
        per = []
        for span in spans:
            span_text = text[span.begin:span.end]
            span_text = [self.stemmer.stem(token.text) for token in tokenize(span_text)]
            if span.tag == EntitySpanPb.Tag.Value("LOC"):
                loc.extend(span_text)
            elif span.tag == EntitySpanPb.Tag.Value("PER"):
                per.extend(span_text)
        return {"loc": loc, "per": per}

    @staticmethod
    def encode_date(date):
        def encode_circular_feature(current_value, max_value):
            sin = np.sin(2*np.pi*current_value/max_value)
            cos = np.cos(2*np.pi*current_value/max_value)
            return sin, cos

        midnight = date.replace(hour=0, minute=0, second=0, microsecond=0)
        seconds_from_midnight = (date - midnight).seconds
        sin_time, cos_time = encode_circular_feature(seconds_from_midnight, 24 * 60 * 60)

        day_of_year = date.timetuple().tm_yday
        days_in_year = date.replace(month=12, day=31).timetuple().tm_yday
        sin_day, cos_day = encode_circular_feature(day_of_year, days_in_year)
        new_features = (sin_time, cos_time, sin_day, cos_day)
        return np.append(np.array(new_features[0]), new_features[1:])

    def save_to_tensorboard(self):
        # NOT WORKING
        writer = SummaryWriter()
        writer.add_embedding(self.vectors, metadata, metadata_header=ClusteredDocument.get_header())
        writer.close()

    def save_clusters(self, output_file_name):
        # NOT WORKING
        clusters = []
        for label, cluster in clusters.items():
            if label == -1:
                continue
            clusters.append([e.to_dict() for e in cluster])
        with open(output_file_name, "w") as w:
            json.dump(clusters, w, ensure_ascii=False, indent=4)

