from keras.models import Sequential, model_from_json
from keras.optimizers import SGD
from model import make_model
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tempfile import TemporaryFile

import base64


iris = load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target)


def serialize_keras_model(model: Sequential) -> dict:
    model_arch = model.to_json()
    temp_file = TemporaryFile()
    model.save_weights(temp_file)
    temp_file.seek(0)
    weights_bytes = temp_file.read()
    weights_string = base64.encodebytes(weights_bytes).decode("utf-8")
    return dict(arch=model_arch, weights=weights_string)


def deserialize_keras_model(serialized: dict) -> Sequential:
    model_arch, weights_string = serialized["arch"], serialized["weights"]
    model: Sequential = model_from_json(model_arch)

    optimizer_kws = dict(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    sgd = SGD(**optimizer_kws)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,
                  metrics=['accuracy'])

    weights_bytes = base64.b64decode(weights_string)
    temp_file = TemporaryFile()
    temp_file.write(weights_bytes)
    model.load_weights(temp_file)
    return model


def test_nn_classifier():
    params = dict(subs_count=3, input_dim=4, dense1_units=16,
                  dropout=0.1, dense2_units=8)
    clf = make_model(method="nn_classifier", params=params)
    clf.fit(x_train, y_train, epochs=50, batch_size=50)

    model = clf.model
    serialized = serialize_keras_model(model)
    model = deserialize_keras_model(serialized)
    clf.model = model

    print(clf.score(x_test, y_test))


def test_logistic_regression():
    clf = make_model(LogisticRegression)
    clf.fit(x_train, y_train)
    print("classifier score:", clf.score(x_test, y_test))

    params = dict(solver="lbfgs", multi_class="auto")
    clf = make_model(LogisticRegression, params=params)
    clf.fit(x_train, y_train)
    score_before = clf.score(x_test, y_test)

    attrs = {
        attr: getattr(clf, attr) for attr in
        filter(lambda name: name.endswith("_") and not name.startswith("_"),
               dir(clf))
    }

    clf = make_model(LogisticRegression, params=params, attrs=attrs)
    score_after = clf.score(x_test, y_test)
    assert score_before == score_after


if __name__ == "__main__":
    test_nn_classifier()
    # test_logistic_regression()
