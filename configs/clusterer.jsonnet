{
    "clustering": {
        "type": "agglomerative",
        "affinity": "precomputed",
        "linkage": "average",
        "distance_threshold": 0.36,
        "n_clusters": null
    },
    "fetching": {
        "embeddings": {
            "aggregation": "concat",
            "keys": ["text_linear_fasttext_embedding", "title_linear_fasttext_embedding"],
            "weights": [1.0, 1.0]
        },
        "keywords_key": "tfidf_keywords",
        "entities_key": "title_slovnet_ner"
    },
    "distances": {
        "fix_hosts": true,
        "hosts_penalty": 5.0,
        "fix_time": false,
        "time_penalty": 4.0
    }
}
