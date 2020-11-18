import json
import itertools
from typing import Optional
from collections import defaultdict, Counter

import numpy as np
from _jsonnet import evaluate_file as jsonnet_evaluate_file
from nltk.stem.snowball import SnowballStemmer
from razdel import tokenize
from scipy.special import expit
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import pairwise_distances
from torch.utils.tensorboard import SummaryWriter

from purano.models import Document
from purano.proto.info_pb2 import EntitySpan as EntitySpanPb

# TODO: agency2vec
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
        self.num2keywords = list()
        self.num2host = list()
        self.num2timestamp = list()

        self.id2num = dict()
        self.keyword2nums = defaultdict(list)

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

    def fetch_info(self,
        start_date: Optional[str]=None,
        end_date: Optional[str]=None,
        sort_by_date: Optional[bool]=False,
        nrows: Optional[int]=None
    ):
        print("Fetching documents from database...")
        embeddings_config = self.config.pop("embeddings")
        entities_key = self.config.pop("entities_key", None)
        keywords_key = self.config.pop("keywords_key", None)

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

            del document.info

            vectors.append(vector)
            self.id2num[document.id] = doc_num
            self.num2doc.append(document)
            self.num2host.append(document.host)
            self.num2timestamp.append(document.date.timestamp())

        self.vectors = np.array(vectors)

    def cluster(self):
        fix_hosts = self.config.pop("fix_hosts", False)
        fix_time = self.config.pop("fix_time", False)
        hosts_penalty = self.config.pop("hosts_penalty", 1.0)

        time_penalty_modifier = self.config.pop("time_penalty", 1.0)
        print("Calculating distances matrix...")
        max_distance = 1.0
        distances = np.full((len(self.num2doc), len(self.num2doc)), max_distance, dtype=np.float32)
        for i, (keyword, doc_nums) in enumerate(self.keyword2nums.items()):
            vectors = self.vectors[doc_nums]
            batch_distances = pairwise_distances(vectors, metric="cosine", n_jobs=1, force_all_finite=False)
            for (l1, g1), (l2, g2) in itertools.product(enumerate(doc_nums), repeat=2):
                distances[g1, g2] = batch_distances[l1, l2]
                if fix_hosts and self.num2host[g1] == self.num2host[g2]:
                    distances[g1, g2] = min(max_distance, distances[g1, g2] * hosts_penalty)
                if fix_time and g1 != g2:
                    time_diff = abs(self.num2timestamp[g1] - self.num2timestamp[g2])
                    hours_shifted = (time_diff / 3600) - 12
                    time_penalty = 1.0 + expit(hours_shifted) * (time_penalty_modifier - 1.0)
                    distances[g1, g2] = min(max_distance, distances[g1, g2] * time_penalty)
 
        print("Running clustering algorithm...")
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

    def save_to_tensorboard(self):
        # NOT WORKING
        writer = SummaryWriter()
        writer.add_embedding(self.vectors, metadata, metadata_header=ClusteredDocument.get_header())
        writer.close()

    def save(self, output_file_name):
        clusters = []
        for label, cluster in self.clusters.items():
            if label == -1:
                continue
            clusters.append({"articles": [self.num2doc[self.id2num[doc_id]].url for doc_id in cluster]})
        with open(output_file_name, "w") as w:
            json.dump(clusters, w, ensure_ascii=False, indent=4)
