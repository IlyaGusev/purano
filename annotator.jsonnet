{
    "steps": [
        #"bert_title",
        "rvs_fasttext_title",
        "rvs_fasttext_text",
        "rvs_elmo_title",
        "rvs_elmo_text"
    ],
    "processors": {
        #"deeppavlov_bert": {
        #    "type": "bert",
        #    "pretrained_model_name_or_path": "./models/rubert_cased_L-12_H-768_A-12_v2"
        #},
        "rvs_fasttext": {
            "type": "fasttext",
            "path": "./models/tayga_none_fasttextcbow_300_10_2019/model.model"
        },
        "rvs_elmo": {
            "type": "elmo",
            "options_file": "./models/ruwikiruscorpora_tokens_elmo_1024_2019/options.json",
            "weight_file": "./models/ruwikiruscorpora_tokens_elmo_1024_2019/model.hdf5",
            "cuda_device": -1
        }
    },
    #"bert_title": {
    #    "processor": "deeppavlov_bert",
    #    "input_field": "title",
    #    "output_field": "title_bert_embedding"
    #},
    "rvs_fasttext_title": {
        "processor": "rvs_fasttext",
        "agg_type": "mean||max",
        "input_field": "title",
        "output_field": "title_rvs_fasttext_embedding"
    },
    "rvs_fasttext_text": {
        "processor": "rvs_fasttext",
        "agg_type": "mean||max",
        "input_field": "text",
        "output_field": "text_rvs_fasttext_embedding",
        "max_tokens_count": 200
    },
    "rvs_elmo_title": {
        "processor": "rvs_elmo",
        "agg_type": "mean||max",
        "input_field": "title",
        "output_field": "title_rvs_elmo_embedding"
    },
    "rvs_elmo_text": {
        "processor": "rvs_elmo",
        "agg_type": "mean||max",
        "input_field": "text",
        "output_field": "text_rvs_elmo_embedding",
        "max_tokens_count": 200
    }
}
