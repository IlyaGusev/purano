import argparse
import copy
import os

import neptune
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from hyperopt import Trials, fmin, hp, tpe

from purano.clusterer.clusterer import Clusterer
from purano.clusterer.metrics import calc_metrics
from purano.readers import parse_tg_jsonl, parse_clustering_markup_tsv

SEARCH_SPACE = hp.pchoice(
    "clustering_type", [(0.999, {
        "type": "agglomerative",
        "distance_threshold": hp.quniform("distance_threshold", 0.01, 0.95, 0.01),
        "linkage": "average",
        "n_clusters": None,
        "affinity": "precomputed"
    }), (0.001, {
        "type": "dbscan",
        "eps": hp.quniform("dbscan_eps", 0.01, 0.8, 0.01),
        "min_samples": hp.quniform("dbscan_min_samples", 1, 20, 1),
        "leaf_size": hp.quniform("dbscan_leaf_size", 2, 50, 2),
        "metric": "precomputed"
    })]
)


def fit_param(
    input_file: str,
    nrows: int,
    sort_by_date: bool,
    start_date: str,
    end_date: str,
    config: str,
    clustering_markup_tsv: str,
    original_jsonl: str,
    neptune_project: str,
    neptune_tags: str
):
    assert input_file.endswith(".db")
    assert os.path.isfile(clustering_markup_tsv)
    assert clustering_markup_tsv.endswith(".tsv")
    assert os.path.isfile(original_jsonl)
    assert original_jsonl.endswith(".jsonl")
    assert os.path.isfile(config)
    assert config.endswith(".jsonnet")
    assert neptune_tags
    neptune_tags = neptune_tags.split(",")

    neptune_api_token = os.getenv("NEPTUNE_API_TOKEN")

    neptune.init(project_qualified_name=neptune_project, api_token=neptune_api_token)

    url2record = {r["url"]: r for r in parse_tg_jsonl(original_jsonl)}
    markup = parse_clustering_markup_tsv(clustering_markup_tsv)

    db_engine = "sqlite:///{}".format(input_file)
    engine = create_engine(db_engine)
    Session = sessionmaker(bind=engine, autoflush=False)
    session = Session()

    clusterer = Clusterer(session, config)
    clusterer.fetch_info(
        start_date,
        end_date,
        sort_by_date,
        nrows
    )

    config_copy = copy.deepcopy(clusterer.config)

    def calc_accuracy(params):
        neptune.create_experiment(
            name="clustering",
            params=params,
            upload_source_files=['configs/*.jsonnet'],
            tags=neptune_tags
        )
        clusterer.config = copy.deepcopy(config_copy)
        clusterer.config["clustering"] = params
        clusterer.config["distances"]["cache_distances"] = True
        clusterer.calc_distances()
        clusterer.cluster()
        labels = clusterer.get_labels()
        clusterer.reset_clusters()
        metrics, _ = calc_metrics(markup, url2record, labels)
        accuracy = metrics["accuracy"]
        f1_score_0 = metrics["0"]["f1-score"]
        f1_score_1 = metrics["1"]["f1-score"]
        neptune.log_metric("accuracy", accuracy)
        neptune.log_metric("f1_score_0", f1_score_0)
        neptune.log_metric("f1_score_1", f1_score_1)
        neptune.stop()
        return -accuracy

    trials = Trials()
    best = fmin(
        fn=calc_accuracy,
        space=SEARCH_SPACE,
        algo=tpe.suggest,
        max_evals=200,
        trials=trials
    )
    print(best)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, default="output/0525_annotated.db")
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--sort-by-date", default=False,  action='store_true')
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--clustering-markup-tsv", type=str, required=True)
    parser.add_argument("--original-jsonl", type=str, required=True)
    parser.add_argument("--neptune-project", type=str, required=True)
    parser.add_argument("--neptune-tags", type=str, required=True)

    args = parser.parse_args()
    fit_param(**vars(args))
