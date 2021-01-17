import csv


def read_markup_tsv(file_name, clean_header=True):
    records = []
    with open(file_name, "r") as r:
        header = tuple(next(r).strip().split("\t"))
        if clean_header:
            header = tuple((h.split(":")[-1] for h in header))
        reader = csv.reader(r, delimiter='\t', quotechar='"')
        for row in reader:
            record = dict(zip(header, row))
            records.append(record)
    return records


def write_markup_tsv(records, file_name, res_prefix="GOLDEN", res_key="quality"):
    with open(file_name, "w") as w:
        writer = csv.writer(w, delimiter='\t', quotechar='"')
        keys = set()
        for record in records:
            keys = keys.union(set(record.keys()))
        keys = list(keys)
        if res_key not in keys:
            keys.append(res_key)
        header = ["INPUT:" + k if k != res_key else res_prefix + ":" + res_key for k in keys]
        writer.writerow(header)
        for record in records:
            row = [record.get(key, "").replace("\n", " ").strip() for key in keys]
            writer.writerow(row)
