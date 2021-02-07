from sklearn.metrics import classification_report


def calc_metrics(markup, url2record, labels):
    not_found_count = 0
    for record in markup:
        first_url = record["first_url"]
        second_url = record["second_url"]
        not_found_in_labels = first_url not in labels or second_url not in labels
        not_found_in_records = first_url not in url2record or second_url not in url2record
        if not_found_in_labels or not_found_in_records:
            not_found_count += 1
            markup.remove(record)
    if not_found_count != 0:
        print("Not found {} pairs from markup".format(not_found_count))

    targets = []
    predictions = []
    errors = []
    for record in markup:
        first_url = record["first_url"]
        second_url = record["second_url"]
        target = int(record["quality"] == "OK")
        prediction = int(labels[first_url] == labels[second_url])
        first = url2record.get(first_url)
        second = url2record.get(second_url)
        targets.append(target)
        predictions.append(prediction)
        if target == prediction:
            continue
        errors.append({
            "target": target,
            "prediction": prediction,
            "first_url": first_url,
            "second_url": second_url,
            "first_title": first["title"],
            "second_title": second["title"],
            "first_text": first["text"],
            "second_text": second["text"]
        })

    metrics = classification_report(targets, predictions, output_dict=True)
    return metrics, errors
