{
    "steps": [
        "fasttext_title",
        "fasttext_title_text",
        "fasttext_title_linear",
        "fasttext_text_linear",
        "ner_slovnet_title",
        "gen_title_encoder",
        "tfidf_keywords",
        "tfidf",
        "labse"
    ],
    "processors": {
        "fasttext": {
            "type": "fasttext",
            "vector_model_path": "./models/fasttext/ru_vectors_v3.bin"
        },
        "fasttext_text_linear": {
            "type": "fasttext",
            "vector_model_path": "./models/fasttext/ru_vectors_v3.bin",
            "torch_model_path": "./models/text2title/ru_ft_text_embedder.pt"
        },
        "fasttext_title_linear": {
            "type": "fasttext",
            "vector_model_path": "./models/fasttext/ru_vectors_v3.bin",
            "torch_model_path": "./models/text2title/ru_ft_title_embedder.pt"
        },
        "ner_slovnet": {
            "type": "ner_slovnet",
            "model_path": "./models/slovnet/slovnet_ner_news_v1.tar",
            "vector_model_path": "./models/slovnet/navec_news_v1_1B_250K_300d_100q.tar"
        },
        "tfidf_keywords": {
            "type": "tfidf_keywords",
            "idfs_vocabulary": "./models/tfidf/ru_idfs.txt",
            "top_k": 15
        },
        "tfidf": {
            "type": "tfidf",
            "idfs_vocabulary": "./models/tfidf/ru_idfs.txt",
            "svd_torch_model_path": "./models/tfidf/svd_matrix.pt"
        },
        "gen_title_encoder": {
            "type": "transformers",
            "pretrained_model_name_or_path": "IlyaGusev/gen_title_tg_bottleneck_encoder",
            "use_gpu": true,
            "do_lower_case": false
        },
        "labse": {
            "type": "transformers",
            "pretrained_model_name_or_path": "sentence-transformers/LaBSE",
            "use_gpu": true,
            "do_lower_case": false
        }
    },
    "fasttext_title": {
        "processor": "fasttext",
        "agg_type": "mean||max||min",
        "input_fields": ["patched_title"],
        "output_field": "title_fasttext_embedding",
        "use_preprocessing": false,
        "max_tokens_count": 100
    },
    "fasttext_text_linear": {
        "processor": "fasttext_text_linear",
        "agg_type": "linear",
        "input_fields": ["patched_text"],
        "output_field": "text_linear_fasttext_embedding",
        "max_tokens_count": 200,
        "use_preprocessing": false
    },
    "fasttext_title_linear": {
        "processor": "fasttext_title_linear",
        "agg_type": "linear",
        "input_fields": ["patched_title"],
        "output_field": "title_linear_fasttext_embedding",
        "max_tokens_count": 100,
        "use_preprocessing": false
    },
    "fasttext_title_text": {
        "processor": "fasttext",
        "agg_type": "mean||max||min",
        "input_fields": ["patched_title", "patched_text"],
        "output_field": "title_text_fasttext_embedding",
        "max_tokens_count": 200,
        "use_preprocessing": false
    },
    "ner_slovnet_title": {
        "processor": "ner_slovnet",
        "input_fields": ["title"],
        "output_field": "title_slovnet_ner"
    },
    "tfidf_keywords": {
        "processor": "tfidf_keywords",
        "input_fields": ["patched_title", "patched_text"],
        "output_field": "tfidf_keywords"
    },
    "tfidf": {
        "processor": "tfidf",
        "input_fields": ["patched_title", "patched_text"],
        "output_field": "tfidf_embedding"
    },
    "gen_title_encoder": {
        "processor": "gen_title_encoder",
        "input_fields": ["text"],
        "output_field": "gen_title_embedding",
        "max_tokens_count": 196,
        "layer": -1,
        "aggregation": "first"
    },
    "labse": {
        "processor": "labse",
        "input_fields": ["title", "text"],
        "output_field": "labse_embedding",
        "max_tokens_count": 100,
        "aggregation": "pooler",
        "normalize": true
    }
}
