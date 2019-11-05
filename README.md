# PuraNo - news annotation and clustering

## Prerequisites
pip install -r requirements.txt

## Installation
[Datasets](datasets.txt)

[Download script](download.sh)


## Scripts
#### [parse_csv](purano/parse_csv.py)

Parse files with comma separated values with news documents from github.com/ods-ai-ml4sg/proj_news_viz and save them in SQLite database.

| Argument      | Default           | Description                                                   |
|:--------------|:------------------|:--------------------------------------------------------------|
| -f, --files   |                   | path to a file/files to parse                                 |
| --db-engine   | sqlite:///news.db | database type and path                                        |

#### [annotate](purano/annotate.py)

Compute different text embeddings based on word2vec/fasttext/ELMo/BERT models. Config defines what models to use and in what way. Embeddings are stored as [protobufs](https://developers.google.com/protocol-buffers) in separate SQLite table column, schema can be found in [info.proto](purano/proto/info.proto). To modify schema simply change [info.proto](purano/proto/info.proto) and run [compile_proto.sh](compile_proto.sh).

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



