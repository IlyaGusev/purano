import os

def parse_dir(directory, ext, parse_file_func, print_interval=None):
    documents_count = 0
    for r, d, f in os.walk(directory):
        for file_name in f:
            file_name = os.path.join(r, file_name)
            if not file_name.endswith(ext):
                continue
            try:
                for record in parse_file_func(file_name):
                    yield record
                    documents_count += 1
                    if print_interval and documents_count % print_interval == 0:
                        print("Parsed {} documents".format(documents_count))
            except Exception as e:
                print(e)
                continue
