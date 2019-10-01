from sqlalchemy import create_engine, Column, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from bert_serving.client import BertClient
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import numpy as np

Base = declarative_base()


class Doc(Base):
    __tablename__ = "docs"
    url = Column(String, primary_key=True)
    text = Column(Text)
    title = Column(Text)
    topics = Column(Text)


class Annotation:
    def __init__(self, doc):
        self.title_embedding = None
        self.doc = doc


def annotate_document(record, bert_client):
    annotation = Annotation(record)
    annotation.title_embedding = bert_client.encode([record.title])[0]
    print("Annotated: {}".format(record.title))
    return annotation


def train_clf(data):
    all_topics = list({annot.doc.topics for annot in data})
    
    def to_features(annotation):
        return annotation.title_embedding
    
    def to_target(annotation):
        return all_topics.index(annotation.doc.topics)

    X = np.array([to_features(annot) for annot in data])
    y = [to_target(annot) for annot in data]
    border = len(data) * 9 // 10
    print(X.shape)
    print(y)
    X_train, X_test = X[:border], X[border:]
    y_train, y_test = y[:border], y[border:]
    clf = MLPClassifier()
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    print(metrics.classification_report(y_test, y_predicted))


def main():
    bert_client = BertClient()

    engine = create_engine("sqlite:///news.db")
    Session = sessionmaker(bind=engine)
    session = Session()
    query = session.query(Doc)
    records = query.limit(1000)
    annotations = []
    for record in records:
        annotations.append(annotate_document(record, bert_client))
    train_clf(annotations) 
    

main()
