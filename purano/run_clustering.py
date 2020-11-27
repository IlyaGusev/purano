import argparse

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pyinstrument import Profiler

from purano.clusterer.clusterer import Clusterer


def cluster(
    input_file: str,
    nrows: int,
    sort_by_date: bool,
    start_date: str,
    end_date: str,
    config: str,
    output_file: str
):
    db_engine = "sqlite:///{}".format(input_file)
    engine = create_engine(db_engine)
    Session = sessionmaker(bind=engine, autoflush=False)
    session = Session()

    clusterer = Clusterer(session, config)
    clusterer.fetch_info(
        start_date,
        end_date,
        sort_by_date,
        nrows
    )
    clusterer.calc_distances()
    clusterer.cluster()
    clusterer.print_clusters()
    clusterer.save(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, default="output/train_annotated.db")
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--sort-by-date", default=False,  action='store_true')
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-file", type=str, default="output/clusters.json")
    parser.add_argument("--perfomance-log", type=str, default="clustering_performance.txt")

    args = parser.parse_args()
    profiler = Profiler()
    profiler.start()
    args = vars(args)
    perfomance_log = args.pop("perfomance_log")
    cluster(**args)
    profiler.stop()

    with open(perfomance_log, "w") as w:
        w.write(profiler.output_text(unicode=True, color=True, show_all=True))
