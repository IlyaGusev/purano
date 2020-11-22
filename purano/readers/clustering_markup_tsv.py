import csv
from collections import defaultdict

def parse_clustering_markup_tsv(file_name):
    markup = defaultdict(dict)
    with open(file_name, "r") as r:
        reader = csv.reader(r, delimiter='\t', quotechar='"')
        header = next(reader)
        for row in reader:
            assert len(header) == len(row)
            record = dict(zip(header, row))
            first_url = record["INPUT:first_url"]
            second_url = record["INPUT:second_url"]
            quality = int(record["OUTPUT:quality"] == "OK")
            markup[(first_url, second_url)] = quality
    return markup

