from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.base import ClassifierMixin, BaseEstimator
from typing import Optional, Union, Callable, Dict


def build_fn(subs_count: int, input_dim: Optional[int]=128,
             dense1_units: Optional[int]=64, dense2_units: Optional[int]=32,
             dense1_kws: Optional[dict]=None, dense2_kws: Optional[dict]=None,
             dense3_kws: Optional[dict]=None, dropout: Optional[float]=0.5,
             optimizer_kws: Optional[dict]=None):

    model = Sequential()
    if dense1_kws is None:
        dense1_kws = dict(activation="relu")
    dense1_kws["input_dim"] = input_dim
    model.add(Dense(dense1_units, **dense1_kws))
    model.add(Dropout(dropout))

    if dense2_kws is None:
        dense2_kws = dict(activation="relu")
    model.add(Dense(dense2_units, **dense2_kws))

    if dense3_kws is None:
        dense3_kws = dict(activation="softmax")
    model.add(Dense(subs_count, **dense3_kws))

    if optimizer_kws is None:
        optimizer_kws = dict(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    sgd = SGD(**optimizer_kws)

    model.compile(loss='categorical_crossentropy', optimizer=sgd,
                  metrics=['accuracy'])
    return model


def make_nn_classifier(**kwargs: dict):
    assert "subs_count" in kwargs, "subject count must be provided"
    model = KerasClassifier(build_fn, **kwargs)
    return model


EstimatorFactory = Callable[[Optional[dict]], BaseEstimator]
EstimatorDeserializer = Callable[[Union[str, bytes]], BaseEstimator]

METHODS: Dict[str, EstimatorFactory] = {
    "nn_classifier": make_nn_classifier
}

DESERIALIZATION_METHODS: Dict[str, EstimatorDeserializer] = {

}


def make_model(method: Optional[Union[str, type]]=None,
               params: Optional[dict]=None,
               deserializer: Optional[Union[EstimatorDeserializer, str]]=None,
               serialized_model: Optional[Union[str, bytes]]=None,
               attrs: Optional[dict]=None) -> BaseEstimator:
    """
    Creates an Estimator and sets parameters and attributes of the estimator.
    Model can be created by:

    1. De-serializing serialized model string
    2. Providing model class and configurations

    Various attributes can be added to model like pre-computed weight matrix,
    regularization method, model specific attributes etc.

    Model parameters should be set using set_params method of estimators

    This function provides functionality to achieve this through attrs
    dictionary. Each entry in attrs is added to created model as a model
    attribute and consumed by model if it is a defined attribute.

    Parameters
    ----------
    method : string, function, optional
        Method for model creation. It can be name of a method defined in
        `METHODS` dictionary in this module or it can be a callable that
        returns instance of model upon call.

    params : dict, optional
        Model parameters. These are set using `set_params` method of Estimator

    deserializer : string, function, optional
        Method for de-serializing model.

    serialized_model : object, optional
        Model in serialized form

    attrs : dict, optional
        Set attributes of created model, typical set of attributes include
        model parameters like regularization method, optimization method,
        model constants like penalty factor. This can also contain values
        like weight matrix of a linear model. Which are actually learned
        values and not parameters.

    Returns
    -------
    model : sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin
        Returns compiled classifier object

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> iris = load_iris()
    >>> x_train, x_test, y_train, y_test = \
    ...         train_test_split(iris.data, iris.target)
    >>> params = dict(subs_count=3, input_dim=4, dense1_units=16,
    ...               dropout=0.1, dense2_units=8)
    >>> clf = make_model(method="nn_classifier", params=params)
    >>> clf.fit(x_train, y_train, epochs=50, batch_size=50)
    >>> clf.score(x_test, y_test)

    >>> from sklearn.linear_model import LogisticRegression
    >>> clf = make_model(method=LogisticRegression)
    >>> clf.fit(x_train, y_train)
    >>> print("classifier score:", clf.score(x_test, y_test))
    >>> # classifier score: 0.9736842105263158

    >>> params = dict(solver="lbfgs", multi_class="auto")
    >>> clf = make_model(method=LogisticRegression, params=params)
    >>> clf.fit(x_train, y_train)
    >>> score_before = clf.score(x_test, y_test)
    >>> attrs = {attr: getattr(clf, attr) for attr in
    ...          ["classes_", "coef_", "intercept_", "n_iter_"]}
    >>> clf = make_model(method=LogisticRegression, params=params, attrs=attrs)
    >>> score_after = clf.score(x_test, y_test)
    >>> assert score_before == score_after, "attributes not set properly"
    """
    if params is None:
        params = {}

    if serialized_model is not None:
        if deserializer in DESERIALIZATION_METHODS:
            dsm = DESERIALIZATION_METHODS[deserializer]
            model: BaseEstimator = dsm(serialized_model)
        elif callable(deserializer):
            model: BaseEstimator = deserializer(serialized_model)
        else:
            model = None
    elif method is not None and method in METHODS:
        model: BaseEstimator = METHODS[method](**params)
    elif method is not None and (issubclass(method, ClassifierMixin)
                                 and issubclass(method, BaseEstimator)):
        model: BaseEstimator = method(**params)
    else:
        model = None

    if model is None:
        raise ValueError("could not create model")

    if attrs is not None:
        for attr, value in attrs.items():
            setattr(model, attr, value)

    return model
