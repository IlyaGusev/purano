from urllib.parse import urlsplit
from datetime import datetime

import scrapy
from scrapy.loader import ItemLoader

from parsers.items import Document


class NewsSpiderConfig:
    def __init__(self, title_path, summary_path, date_path, date_format,
                 text_path, topics_path, authors_path):
        self.title_path = title_path
        self.summary_path = summary_path
        self.date_path = date_path
        self.date_format = date_format
        self.text_path = text_path
        self.topics_path = topics_path
        self.authors_path = authors_path


class NewsSpider(scrapy.Spider):
    def __init__(self, *args, **kwargs):
        assert "until_date" in kwargs
        assert self.config
        kwargs["until_date"] = datetime.strptime(kwargs["until_date"], "%d.%m.%Y").date()
        super().__init__(*args, **kwargs)

    def parse_document(self, response):
        url = response.url
        base_edition = urlsplit(self.start_urls[0])[1]
        edition = urlsplit(url)[1]

        loader = ItemLoader(item=Document(), response=response)
        loader.add_value("url", url)
        loader.add_value("edition", "-" if edition == base_edition else edition)
        loader.add_xpath("title", self.config.title_path)
        loader.add_xpath("summary", self.config.summary_path)
        loader.add_xpath("date", self.config.date_path)
        loader.add_xpath("text", self.config.text_path)
        loader.add_xpath("topics", self.config.topics_path)
        loader.add_xpath("authors", self.config.authors_path)

        yield loader.load_item()

    def process_title(self, title):
        return title.replace("\xa0", " ")

    def process_text(self, paragraphs):
        text = " ".join([p.strip() for p in paragraphs if p.strip()])
        text = text.replace("\xa0", " ").replace(" . ", ". ").strip()
        return text

    def process_summary(self, summary):
        return summary.replace("\xa0", " ").strip()
