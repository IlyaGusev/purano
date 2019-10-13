import argparse
import json
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from catboost import CatBoostClassifier

from purano.models import Document, Info


def train_clf(annotations, field):
    annotations = list(annotations)
    docs = [annot.document for annot in annotations]
    all_topics = list({doc.topics for doc in docs})

    def to_features(annotation):
        return np.array(getattr(annotation.get_info(), field))

    def to_target(doc):
        return all_topics.index(doc.topics)

    X = np.array([to_features(annot) for annot in annotations])
    y = [to_target(doc) for doc in docs]
    val_border = X.shape[0] * 8 // 10
    test_border = X.shape[0] * 9 // 10
    X_train, X_val, X_test = X[:val_border], X[val_border:test_border], X[test_border:]
    y_train, y_val, y_test = y[:val_border], y[val_border:test_border], y[test_border:]
    #clf = MLPClassifier()
    clf = CatBoostClassifier(iterations=5000,
                             loss_function="MultiClass",
                             task_type="GPU",
                             verbose=True)
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    print(X_train.shape)
    #clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    print(metrics.classification_report(y_test, y_predicted))


def main(db_engine, nrows, sort_by_date, start_date, end_date, agency_id, field):
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
    annotations = query.limit(nrows) if nrows else query.all()
    train_clf(annotations, field)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-engine", type=str, default="sqlite:///news.db")
    parser.add_argument("--sort-by-date", default=False,  action='store_true')
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--agency-id", type=int, default=None)
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--field", type=str, required=True)

    args = parser.parse_args()
    main(**vars(args))
