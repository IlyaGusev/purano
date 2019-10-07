import argparse
import json
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from catboost import CatBoostClassifier

from models import Document, Info


def train_clf(annotations):
    annotations = list(annotations)
    docs = [annot.document for annot in annotations]
    all_topics = list({doc.topics for doc in docs})

    def to_features(annotation):
        return np.array(annotation.get_info().title_bert_embedding)

    def to_target(doc):
        return all_topics.index(doc.topics)

    X = np.array([to_features(annot) for annot in annotations])
    y = [to_target(doc) for doc in docs]
    val_border = X.shape[0] * 8 // 10
    test_border = X.shape[0] * 9 // 10
    X_train, X_val, X_test = X[:val_border], X[val_border:test_border], X[test_border:]
    y_train, y_val, y_test = y[:val_border], y[val_border:test_border], y[test_border:]
    #clf = MLPClassifier()
    clf = CatBoostClassifier(iterations=500,
                             depth=5,
                             loss_function='MultiClass',
                             verbose=True)
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    y_predicted = clf.predict(X_test)
    print(metrics.classification_report(y_test, y_predicted))


def main(db_engine, nrows):
    engine = create_engine(db_engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    query = session.query(Info)
    annotations = query.limit(nrows) if nrows else query.all()
    train_clf(annotations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-engine", type=str, default="sqlite:///news.db")
    parser.add_argument("--nrows", type=int, default=None)

    args = parser.parse_args()
    main(**vars(args))
