{
    "steps": [
        "fasttext_title",
        "fasttext_title_text",
        "xlmroberta_title",
        "elmo_title"
    ],
    "processors": {
        "fasttext": {
            "type": "fasttext",
            "path": "./models/ru_vectors_v3.bin"
        },
        "xlmroberta": {
            "type": "transformers",
            "pretrained_model_name_or_path": "xlm-roberta-base"
        },
        "rvs_elmo": {
            "type": "elmo",
            "options_file": "./models/ruwikiruscorpora_tokens_elmo_1024_2019/options.json",
            "weight_file": "./models/ruwikiruscorpora_tokens_elmo_1024_2019/model.hdf5",
            "cuda_device": 0
        },
        #"ner_client": {
        #    "type": "ner_client",
        #    "port": 8889,
        #    "ip": "0.0.0.0"
        #}
    },
    "elmo_title": {
        "processor": "rvs_elmo",
        "input_fields": ["title"],
        "output_field": "title_elmo_embedding",
        "max_tokens_count": 100
    },
    "fasttext_title": {
        "processor": "fasttext",
        "agg_type": "mean||max",
        "input_fields": ["title"],
        "output_field": "title_fasttext_embedding",
        "max_tokens_count": 100
    },
    "fasttext_title_text": {
        "processor": "fasttext",
        "agg_type": "mean||max",
        "input_fields": ["title", "text"],
        "output_field": "title_text_fasttext_embedding",
        "max_tokens_count": 200
    },
    "xlmroberta_title": {
        "processor": "xlmroberta",
        "input_fields": ["title"],
        "output_field": "title_xlmroberta_embedding",
        "max_tokens_count": 100,
        "layer": -1
    }
}
