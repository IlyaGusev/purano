{
    "clustering_type": "agglomerative",
    "clustering_params": {
        "affinity": "precomputed",
        "linkage": "average",
        "distance_threshold": 0.32,
        "n_clusters": null
    },
    "embeddings": {
        "aggregation": "concat",
        "keys": ["text_linear_fasttext_embedding", "title_linear_fasttext_embedding"],
        "weights": [1.0, 1.0]
    },
    "entities_key": "title_slovnet_ner",
    "hosts_penalty": 5.0,
    "fix_hosts": true
}
