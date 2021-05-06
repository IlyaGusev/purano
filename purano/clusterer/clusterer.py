import json
import itertools
from typing import Optional
from collections import defaultdict

import numpy as np
from _jsonnet import evaluate_file as jsonnet_evaluate_file
from nltk.stem.snowball import SnowballStemmer
from scipy.special import expit
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import pairwise_distances

from purano.models import Document
from purano.proto.info_pb2 import EntitySpan as EntitySpanPb

# TODO: agency2vec
# TODO: extracting geo

CLUSTERINGS = {
    "agglomerative": AgglomerativeClustering,
    "dbscan": DBSCAN
}


class Clusterer:
    def __init__(self, db_session, config_path: str):
        self.config = json.loads(jsonnet_evaluate_file(config_path))
        self.db_session = db_session
        self.vectors = None

        self.num2doc = list()
        self.num2entities = list()
        self.num2keywords = list()
        self.num2host = list()
        self.num2timestamp = list()
        self.doc_count = 0

        self.id2num = dict()
        self.keyword2nums = defaultdict(list)

        self.distances = None

        self.labels = dict()
        self.clusters = defaultdict(list)

    def build_final_embedding(self, info, embeddings_config):
        aggregation = embeddings_config["aggregation"]
        keys = embeddings_config["keys"]
        weights = embeddings_config["weights"]
        assert len(keys) == len(weights)
        vectors = []
        for weight, key in zip(weights, keys):
            vector = np.array(info[key])
            vector *= weight
            vectors.append(vector)
        if aggregation == "concat":
            return np.concatenate(vectors, axis=0)
        assert len({len(v) for v in vectors}) == 1, "Embeddings have different dimensions"
        vectors = np.array(vectors)
        if aggregation == "max":
            return np.max(vectors, axis=1)
        if aggregation == "min":
            return np.min(vectors, axis=1)
        assert aggregation == "sum", "Unknown aggregation: {}".format(aggregation)
        return np.sum(vectors, axis=1)

    def fetch_info(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        sort_by_date: Optional[bool] = False,
        nrows: Optional[int] = None
    ):
        print("Fetching documents from database...")
        config = self.config["fetching"]
        embeddings_config = config["embeddings"]
        entities_key = config.pop("entities_key", None)
        keywords_key = config.pop("keywords_key", None)

        query = self.db_session.query(Document)
        if start_date:
            query = query.filter(Document.date > start_date)
        if end_date:
            query = query.filter(Document.date < end_date)
        if sort_by_date:
            query = query.order_by(Document.date)
        query = query.join(Document.info)
        documents = list(query.limit(nrows)) if nrows else list(query.all())

        vectors = []
        for doc_num, document in enumerate(documents):
            if doc_num % 1000 == 0:
                print("Fetched {} documents".format(doc_num))
            info = document.info
            vector = self.build_final_embedding(info, embeddings_config)

            if entities_key:
                entities_pb = info[entities_key]
                entities = self.collect_entities(entities_pb, document.title)
                self.num2entities.append(entities)

            if keywords_key:
                keywords_pb = info[keywords_key]
                keywords = list(keywords_pb)
                self.num2keywords.append(keywords)
                for keyword in keywords:
                    self.keyword2nums[keyword].append(doc_num)
            else:
                self.keyword2nums[""].append(doc_num)

            del document.info

            vectors.append(vector)
            self.id2num[document.id] = doc_num
            self.num2doc.append(document)
            self.num2host.append(document.host)
            self.num2timestamp.append(document.date.timestamp())

        self.doc_count = len(self.num2doc)
        self.vectors = np.array(vectors)

    def _calc_distances(self, start_index, end_index):
        print("Calculating distances matrix for {} to {}...".format(start_index, end_index))
        config = self.config["distances"]
        fix_hosts = config.pop("fix_hosts", False)
        fix_time = config.pop("fix_time", False)
        hosts_penalty = config.pop("hosts_penalty", 1.0)
        time_penalty_modifier = config.pop("time_penalty", 1.0)

        max_distance = 1.0
        window_size = end_index - start_index
        distances = np.full((window_size, window_size), max_distance, dtype=np.float64)
        for i, (_, doc_nums) in enumerate(self.keyword2nums.items()):
            doc_nums = [num for num in doc_nums if start_index <= num < end_index]
            if not doc_nums:
                continue
            vectors = self.vectors[doc_nums]
            batch_distances = pairwise_distances(
                vectors,
                metric="cosine",
                n_jobs=1, # For minimal memory consumption
                force_all_finite=False # For performance
            )
            # l - local, g - global, b - batch-level
            for (l1, g1), (l2, g2) in itertools.product(enumerate(doc_nums), repeat=2):
                b1 = g1 - start_index
                b2 = g2 - start_index
                distances[b1, b2] = batch_distances[l1, l2]
                if fix_hosts and self.num2host[g1] == self.num2host[g2]:
                    distances[b1, b2] = min(max_distance, distances[b1, b2] * hosts_penalty)
                if fix_time and g1 != g2:
                    time_diff = abs(self.num2timestamp[g1] - self.num2timestamp[g2])
                    hours_shifted = (time_diff / 3600) - 12
                    time_penalty = 1.0 + expit(hours_shifted) * (time_penalty_modifier - 1.0)
                    distances[b1, b2] = min(max_distance, distances[b1, b2] * time_penalty)
        return distances

    def cluster(self):
        print("Running clustering algorithm...")
        config = self.config["clustering"]
        clustering_type = config.pop("type")
        window_size = config.pop("window_size")
        intersection_size = config.pop("intersection_size")

        self.labels = dict()
        self.clusters = defaultdict(list)
        old_labels_to_new = dict()
        max_label = 0

        start_index = 0
        end_index = min(window_size, self.doc_count)
        while start_index < self.doc_count and start_index < end_index:
            clustering = CLUSTERINGS[clustering_type](**config)
            distances = self._calc_distances(start_index, end_index)
            labels = clustering.fit_predict(distances)
            labels = [label + max_label if label != -1 else label for label in labels]
            max_label = max(labels)

            for doc_num in range(start_index, end_index):
                doc_id = self.num2doc[doc_num].id
                label_num = doc_num - start_index
                label = labels[label_num]

                old_label = self.labels.get(doc_id, None)
                if old_label is not None:
                    old_labels_to_new[old_label] = label

                self.labels[doc_id] = label
                self.clusters[label].append(doc_id)

            noisy_docs = self.clusters.pop(-1, tuple())
            for doc_id in noisy_docs:
                max_label += 1
                self.labels[doc_id] = max_label
                self.clusters[max_label].append(doc_id)

            if end_index == self.doc_count:
                break
            start_index += window_size - intersection_size
            end_index = min(start_index + window_size, self.doc_count)
            if end_index == self.doc_count:
                start_index = max(end_index - window_size, 0)

        if not old_labels_to_new:
            return

        # Compact labels by transitivity
        for old_label, new_label in old_labels_to_new.items():
            newer_label = new_label
            while newer_label is not None:
                newest_label = newer_label
                newer_label = old_labels_to_new.get(newer_label, None)
            assert newest_label is not None
            old_labels_to_new[old_label] = newest_label

        for doc in self.num2doc:
            doc_id = doc.id
            label = self.labels[doc_id]
            new_label = old_labels_to_new.get(label, label)
            self.labels[doc_id] = new_label

        for old_label, new_label in old_labels_to_new.items():
            self.clusters[new_label].extend(self.clusters.pop(old_label))

    def get_labels(self):
        labels = dict()
        for doc_id, label in self.labels.items():
            document = self.num2doc[self.id2num[doc_id]]
            labels[document.url] = label
        return labels

    def reset_clusters(self):
        self.labels = dict()
        self.clusters = defaultdict(list)

    def print_clusters(self, n=5):
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
        stemmer = SnowballStemmer("russian")
        for span in spans:
            span_text = text[span.begin:span.end]
            span_text = [stemmer.stem(token) for token in span_text.split(" ")]
            if span.tag == EntitySpanPb.Tag.Value("LOC"):
                loc.extend(span_text)
            elif span.tag == EntitySpanPb.Tag.Value("PER"):
                per.extend(span_text)
        return {"loc": loc, "per": per}

    def save(self, output_file_name):
        clusters = []
        for label, cluster in self.clusters.items():
            if label == -1:
                continue
            cluster_urls = [self.num2doc[self.id2num[doc_id]].url for doc_id in cluster]
            clusters.append({"articles": cluster_urls})
        with open(output_file_name, "w") as w:
            json.dump(clusters, w, ensure_ascii=False, indent=4)
