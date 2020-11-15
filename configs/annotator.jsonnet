{
    "steps": [
        "fasttext_title",
        "fasttext_title_text",
        "fasttext_title_linear",
        "fasttext_text_linear",
        "ner_slovnet_title"
    ],
    "processors": {
        "fasttext": {
            "type": "fasttext",
            "vector_model_path": "./models/fasttext/ru_vectors_v3.bin"
        },
        "fasttext_text_linear": {
            "type": "fasttext",
            "vector_model_path": "./models/fasttext/ru_vectors_v3.bin",
            "torch_model_path": "./models/fasttext/ru_text_embedder_v1.pt"
        },
        "fasttext_title_linear": {
            "type": "fasttext",
            "vector_model_path": "./models/fasttext/ru_vectors_v3.bin",
            "torch_model_path": "./models/fasttext/ru_title_embedder_v1.pt"
        },
        "ner_slovnet": {
            "type": "ner_slovnet",
            "model_path": "./models/slovnet/slovnet_ner_news_v1.tar",
            "vector_model_path": "./models/slovnet/navec_news_v1_1B_250K_300d_100q.tar"
        }
    },
    "fasttext_title": {
        "processor": "fasttext",
        "agg_type": "mean||max||min",
        "input_fields": ["title"],
        "output_field": "title_fasttext_embedding",
        "max_tokens_count": 100
    },
    "fasttext_text_linear": {
        "processor": "fasttext_text_linear",
        "agg_type": "linear",
        "input_fields": ["text"],
        "output_field": "text_linear_fasttext_embedding",
        "max_tokens_count": 200
    },
    "fasttext_title_linear": {
        "processor": "fasttext_title_linear",
        "agg_type": "linear",
        "input_fields": ["title"],
        "output_field": "title_linear_fasttext_embedding",
        "max_tokens_count": 100
    },
    "fasttext_title_text": {
        "processor": "fasttext",
        "agg_type": "mean||max||min",
        "input_fields": ["title", "text"],
        "output_field": "title_text_fasttext_embedding",
        "max_tokens_count": 200
    },
    "ner_slovnet_title": {
        "processor": "ner_slovnet",
        "input_fields": ["title"],
        "output_field": "title_slovnet_ner"
    }
}
