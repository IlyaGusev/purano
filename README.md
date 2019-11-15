# PuraNo - news annotation and clustering

## Prerequisites
pip install -r requirements.txt

## Installation
[Datasets](datasets.txt)

[Download script](download.sh)


## Scripts
#### [parse_csv](purano/parse_csv.py)

Parse files with comma separated values with news documents from [proj_news_viz](github.com/ods-ai-ml4sg/proj_news_viz) and save them in SQLite database.

Example:
```
python -m purano.parse_csv -f ~/datasets/interfax_20160101_20191015.csv --start-date 2019-01-01 --end-date 2019-02-01
```

| Argument      | Default           | Description                                                   |
|:--------------|:------------------|:--------------------------------------------------------------|
| -f, --files   |                   | path to a file/files to parse                                 |
| --db-engine   | sqlite:///news.db | database type and path                                        |

#### [annotate](purano/annotate.py)

Compute different text embeddings based on word2vec/fasttext/ELMo/BERT models. Config defines what models to use and in what way. Embeddings are stored as [protobufs](https://developers.google.com/protocol-buffers) in separate SQLite table column, schema can be found in [info.proto](purano/proto/info.proto). To modify schema simply change [info.proto](purano/proto/info.proto) and run [compile_proto.sh](compile_proto.sh).

Example:
```
python -m purano.annotate --sort-by-date --start-date 2019-01-01 --end-date 2019-02-01 --config annotator.jsonnet --reannotate
```

| Argument        | Default           | Description                                                   |
|:----------------|:------------------|:--------------------------------------------------------------|
| --config        |                   | path to jsonnet annotator config, [annotator.jsonnet](annotator.jsonnet) can be used as default |
| --batch-size    | 128               | batch size                                                    |
| --db-engine     | sqlite:///news.db | database type and path                                        |
| --reannotate    | False             | whether to rewrite embeddings or skip document                |
| --sort-by-date  | False             | whether to sort documents by date before annotation           |
| --start-date    | None              | filter documents by start date                                |
| --end-date      | None              | filter documents by end date                                  |
| --agency-id     | None              | filter documents by agency id                                 |
| --nrows         | None              | limit number of documents                                     |

#### [train_clf](purano/train_clf.py)

Train a rubric classifier based on the computed embeddings. It is mainly used to evaluate embeddings quality.

Example:
```
python -m purano.train_clf --sort-by-date --start-date 2019-01-01 --end-date 2019-02-01 --agency-id 1 --field title_rvs_elmo_embedding
```

| Argument        | Default           | Description                                                   |
|:----------------|:------------------|:--------------------------------------------------------------|
| --db-engine     | sqlite:///news.db | database type and path                                        |
| --sort-by-date  | False             | whether to sort documents by date before annotation           |
| --start-date    | None              | filter documents by start date                                |
| --end-date      | None              | filter documents by end date                                  |
| --agency-id     | None              | filter documents by agency id                                 |
| --nrows         | None              | limit number of documents                                     |
| --field         |                   | embedding type from proto schema                              |
| --clf-type      | mlp               | classifier type ("mlp" or "catboost")                         |
| --catboost-iterations | 200         | catboost training iterations                                  |
| --catboost-device     | CPU         | catboost device                                               |
| --val-part      | 0.1               | val part                                                      |
| --test-part     | 0.1               | test part                                                     |


#### [train_clf](purano/cluster.py)

Embedding-based clustering of news from different sources.

Example:
```
python -m purano.cluster --sort-by-date --start-date 2019-01-01 --end-date 2019-02-01 --agency-id 1 --field title_text_rvs_elmo_embedding --clustering-type agglomerative
```

| Argument        | Default           | Description                                                   |
|:----------------|:------------------|:--------------------------------------------------------------|
| --db-engine     | sqlite:///news.db | database type and path                                        |
| --sort-by-date  | False             | whether to sort documents by date before annotation           |
| --start-date    | None              | filter documents by start date                                |
| --end-date      | None              | filter documents by end date                                  |
| --agency-id     | None              | filter documents by agency id                                 |
| --nrows         | None              | limit number of documents                                     |
| --field         |                   | embedding type from proto schema                              |
| --clustering-type      |               | clastering type ("agglomerative" or "dbscan")                         |


