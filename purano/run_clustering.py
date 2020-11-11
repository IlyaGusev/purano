import argparse

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pyinstrument import Profiler

from purano.clusterer.clusterer import Clusterer


def cluster(
    db_engine,
    nrows,
    sort_by_date,
    start_date,
    end_date,
    config,
    output_file_name
):
    engine = create_engine(db_engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    clusterer = Clusterer(session, config)
    clusterer.fetch_embeddings(
        start_date,
        end_date,
        sort_by_date,
        nrows
    )
    clusterer.cluster()
    clusterer.print_clusters()
    #clusters = defaultdict(list)
    #for meta in metadata:
    #    clusters[meta.cluster].append(meta)
    #print_clusters_info(clusters)
    #save_clusters(clusters, output_file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-engine", type=str, default="sqlite:///news.db")
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--sort-by-date", default=False,  action='store_true')
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-file-name", type=str, default="clustering.json")

    args = parser.parse_args()
    profiler = Profiler()
    profiler.start()
    cluster(**vars(args))
    profiler.stop()
    print(profiler.output_text(unicode=True, color=True))
