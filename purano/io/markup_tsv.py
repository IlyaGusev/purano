import csv


def read_markup_tsv(file_name, clean_header=True):
    with open(file_name, "r") as r:
        header = tuple(next(r).strip().split("\t"))
        if clean_header:
            header = tuple((h.split(":")[-1] for h in header))
        reader = csv.reader(r, delimiter='\t', quotechar='"')
        records = [dict(zip(header, row)) for row in reader]
        clean_fields = ("first_title", "second_title", "first_text", "second_text")
        for record in records:
            for field in clean_fields:
                if field not in record:
                    continue
                record[field] = record.pop(field).strip().replace("\xa0", " ")
    return records


def clear_field(field):
    if isinstance(field, str):
        return field.replace("\xa0", " ").replace("\n", " ").strip()
    if isinstance(field, list) and field:
        if isinstance(field[0], str):
            return [clear_field(f) for f in field]
    return field


def write_markup_tsv(
    records,
    file_name,
    res_prefix="",
    res_key="quality",
    input_prefix="",
    ordered_keys=None
):
    with open(file_name, "w") as w:
        writer = csv.writer(w, delimiter='\t', quotechar='"')

        keys = set()
        for record in records:
            keys = keys.union(set(record.keys()))
        keys = list(keys)
        if res_key not in keys:
            keys.append(res_key)

        if ordered_keys:
            keys = ordered_keys
        header = [input_prefix + k if k != res_key else res_prefix + k for k in keys]
        writer.writerow(header)
        for record in records:
            row = [clear_field(record.get(key, "")) for key in keys]
            writer.writerow(row)
