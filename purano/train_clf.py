import argparse
import json

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from catboost import CatBoostClassifier

from purano.models import Document, Info


def train_clf_inner(annotations, field, clf_type, catboost_iterations, catboost_device, val_part, test_part):
    annotations = list(annotations)
    docs = [annot.document for annot in annotations]
    all_topics = list({doc.topics for doc in docs})

    def to_features(annotation):
        return np.array(getattr(annotation.get_info(), field))

    X = np.array([to_features(annot) for annot in annotations])
    y = [all_topics.index(doc.topics) for doc in docs]
    print("X shape: {}, y length: {}".format(X.shape, len(y)))

    val_border = int(X.shape[0] * (1.0 - test_part - val_part))
    test_border = int(X.shape[0] * (1.0 - test_part))
    X_train, X_val, X_test = X[:val_border], X[val_border:test_border], X[test_border:]
    y_train, y_val, y_test = y[:val_border], y[val_border:test_border], y[test_border:]

    if clf_type == "mlp":
        clf = MLPClassifier()
        clf.fit(X_train, y_train)
    elif clf_type == "catboost":
        clf = CatBoostClassifier(iterations=catboost_iterations,
                                 loss_function="MultiClass",
                                 task_type=catboost_device,
                                 verbose=True)
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    y_predicted = clf.predict(X_test)
    print(metrics.classification_report(y_test, y_predicted))


def train_clf(db_engine,
              nrows,
              sort_by_date,
              start_date,
              end_date,
              agency_id,
              field,
              clf_type,
              catboost_iterations,
              catboost_device,
              val_part,
              test_part):
    engine = create_engine(db_engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    query = session.query(Info)
    query = query.join(Info.document)
    if start_date:
        query = query.filter(Document.date > start_date)
    if end_date:
        query = query.filter(Document.date < end_date)
    if agency_id:
        query = query.filter(Document.agency_id == agency_id)
    if sort_by_date:
        query = query.order_by(Document.date)
    query = query.filter(Document.topics != None)
    annotations = query.limit(nrows) if nrows else query.all()
    train_clf_inner(annotations, field, clf_type, catboost_iterations,
                    catboost_device, val_part, test_part)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-engine", type=str, default="sqlite:///news.db")
    parser.add_argument("--sort-by-date", default=False,  action='store_true')
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--agency-id", type=int, default=None)
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--field", type=str, required=True)
    parser.add_argument("--clf-type", type=str, default="mlp", choices=["mlp", "catboost"])
    parser.add_argument("--catboost-iterations", type=int, default=200)
    parser.add_argument("--catboost-device", type=str, default="CPU")
    parser.add_argument("--val-part", type=float, default=0.1)
    parser.add_argument("--test-part", type=float, default=0.1)

    args = parser.parse_args()
    train_clf(**vars(args))
