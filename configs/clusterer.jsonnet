{
    "clustering_type": "agglomerative",
    "clustering_params": {
        "affinity": "precomputed",
        "linkage": "average",
        "distance_threshold": 0.36,
        "n_clusters": null
    },
    "embeddings": {
        "aggregation": "concat",
        "keys": ["text_linear_fasttext_embedding", "title_linear_fasttext_embedding"],
        "weights": [1.0, 1.0]
    },
    "entities_key": "title_slovnet_ner",
    "keywords_key": "tfidf_keywords",
    "hosts_penalty": 5.0,
    "fix_hosts": true,
    "fix_time": false,
    "time_penalty": 4.0
}
