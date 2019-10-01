import pandas as pd
import sqlite3


INTERFAX_PATH = "/media/yallen/My Passport/Datasets/News/interfax.csv"
GAZETA_PATH = "/media/yallen/My Passport/Datasets/News/gazeta.csv"


def fix_line_feed(input_file_name, output_file_name):
    with open(input_file_name, "r") as r, open(output_file_name, "w") as w:
        for line in r:
            line = line.replace("\\n", "\\\\n")
            w.write(line)
            

def process_parser_data(file_name):
    dataset = pd.read_csv(
        file_name, sep=',', quotechar='"', escapechar='\\',
        encoding='utf-8', error_bad_lines=False, header=0,
        verbose=False, keep_date_col=True, index_col=False)
    dataset = dataset[["date", "url", "edition", "title", "text", "authors", "topics"]]
    dataset["date"] = pd.to_datetime(dataset["date"])
    dataset["text"] = dataset["text"].apply(lambda x: x.replace("\\n", "\n"))
    dataset["edition"] = dataset["edition"].apply(lambda x: None if x == "-" else x)
    print(dataset.info())
    return dataset


def main():
    fix_line_feed(INTERFAX_PATH, INTERFAX_PATH + ".fixed")
    interfax_dataset = process_parser_data(INTERFAX_PATH + ".fixed")
    print(interfax_dataset.iloc[1]["text"])
    
    conn = sqlite3.connect("news.db")
    interfax_dataset.to_sql("docs", conn, if_exists='append')
    conn.close()

main()