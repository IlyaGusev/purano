{
    "steps": [
        "fasttext_title",
        "fasttext_title_text",
        "ner_slovnet_title"
#        "xlmroberta_title",
#        "elmo_title"
    ],
    "processors": {
        "fasttext": {
            "type": "fasttext",
            "vector_model_path": "./models/fasttext/ru_vectors_v3.bin"
        },
#        "xlmroberta": {
#            "type": "transformers",
#            "pretrained_model_name_or_path": "xlm-roberta-base"
#        },
#        "rvs_elmo": {
#            "type": "elmo",
#            "options_file": "./models/ruwikiruscorpora_tokens_elmo_1024_2019/options.json",
#            "weight_file": "./models/ruwikiruscorpora_tokens_elmo_1024_2019/model.hdf5",
#            "cuda_device": 0
#        },
        "ner_slovnet": {
            "type": "ner_slovnet",
            "model_path": "./models/slovnet/slovnet_ner_news_v1.tar",
            "vector_model_path": "./models/slovnet/navec_news_v1_1B_250K_300d_100q.tar"
        }
    },
#    "elmo_title": {
#        "processor": "rvs_elmo",
#        "input_fields": ["title"],
#        "output_field": "title_elmo_embedding",
#        "max_tokens_count": 100
#    },
    "fasttext_title": {
        "processor": "fasttext",
        "agg_type": "mean||max||min",
        "input_fields": ["title"],
        "output_field": "title_fasttext_embedding",
        "max_tokens_count": 100
    },
    "fasttext_title_text": {
        "processor": "fasttext",
        "agg_type": "mean||max||min",
        "input_fields": ["title", "text"],
        "output_field": "title_text_fasttext_embedding",
        "max_tokens_count": 200
    },
#    "xlmroberta_title": {
#        "processor": "xlmroberta",
#        "input_fields": ["title"],
#        "output_field": "title_xlmroberta_embedding",
#        "max_tokens_count": 100,
#        "layer": -1
#    },
    "ner_slovnet_title": {
        "processor": "ner_slovnet",
        "input_fields": ["title"],
        "output_field": "title_slovnet_ner"
    }
}
