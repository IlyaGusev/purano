import unittest
import os
from datetime import datetime

from purano.readers.tg_html import parse_tg_html_dir
from purano.readers.tg_jsonl import parse_tg_jsonl_dir
from purano.readers.csv import parse_csv_dir

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_DATA_DIR = os.path.join(TESTS_DIR, "data")

def assert_document(case, document):
    url = document.get("url")
    host = document.get("host")
    title = document.get("title")
    text = document.get("text")
    date = document.get("date")
    case.assertIsNotNone(url)
    case.assertIsInstance(url, str)
    case.assertIsNotNone(date)
    case.assertIsInstance(date, datetime)
    case.assertIsNotNone(host)
    case.assertIsInstance(host, str)
    case.assertIsNotNone(title)
    case.assertIsInstance(title, str)
    case.assertLess(len(title), 300)
    case.assertIsNotNone(text)
    case.assertIsInstance(text, str)
    case.assertLess(len(text), 10000)
    case.assertEqual(url[:4], "http")
    case.assertIn(host, url)


class TestTgHtmlReader(unittest.TestCase):
    def test_parse_tg_html_dir(self):
        documents = []
        for document in parse_tg_html_dir(TESTS_DATA_DIR):
            assert_document(self, document)
            documents.append(document)
        self.assertEqual(len(documents), 2)


class TestTgJsonlReader(unittest.TestCase):
    def test_parse_tg_jsonl_dir(self):
        documents = []
        for document in parse_tg_jsonl_dir(TESTS_DATA_DIR):
            assert_document(self, document)
            documents.append(document)
        self.assertEqual(len(documents), 2)

class TestCsvReader(unittest.TestCase):
    def test_parse_csv_dir(self):
        documents = []
        for document in parse_csv_dir(TESTS_DATA_DIR):
            assert_document(self, document)
            documents.append(document)
        self.assertEqual(len(documents), 2)
