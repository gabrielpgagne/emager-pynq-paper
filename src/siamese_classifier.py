import numpy as np

from emager_py import transforms, emager_redis
from emager_py.data_processing import cosine_similarity

class CosineSimilarity:
    def __init__(self, num_classes: int | None = None, dims: int | None = None):
        """
        Create a cosine similarity classifier.
        """
        super().__init__()

        if num_classes is not None:
            self.features = np.zeros((num_classes, dims))
            self.n_samples = np.zeros(num_classes)
        else:
            self.features = None
            self.n_samples = None

    def __cosine_similarity(self, X, labels: bool):
        dists = cosine_similarity(X, self.features, False)
        if labels:
            return np.argmax(dists, axis=1)
        return dists

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the similarity classifier.

        Args:
            X : the features of shape (n_samples, n_features)
            y: the labels of shape (n_samples,)
        """
        if self.features is None:
            self.features = np.zeros((len(np.unique(y)), X.shape[1]))
            self.n_samples = np.zeros(len(np.unique(y)))

        tmp_features = self.features
        for i in range(len(self.features)):
            tmp_features[i] *= self.n_samples[i]

        for c in np.unique(y):
            c_labels = (y == c).squeeze()
            self.n_samples[c] += len(c_labels)
            x_ok = X[c_labels]
            tmp_features[c] += np.sum(x_ok, axis=0)

        for i in range(len(self.features)):
            if self.n_samples[i] == 0:
                continue
            self.features = tmp_features / self.n_samples[i]

    def predict(self, X):
        return self.__cosine_similarity(X, True)

    def predict_proba(self, X):
        dists = self.__cosine_similarity(X, False)
        return (dists + 1) / 2.0  # scale [-1, 1] to [0, 1]


class DummyFinnAccel:
    def __init__(self):
        pass

    def execute(self, x):
        return x


class EmgSCNNWrapper:
    def __init__(self, model, classifier, input_shape):
        """The SCNN model wrapper. It includes an EMGSCNN model and a classifier."""
        self.model = model
        self.input_shape = (-1, *input_shape)
        self.attach_classifier(classifier)

    def attach_classifier(self, classifier):
        """Attach an estimator to the model for classification. Required for `self.test_step()`

        Args:
            classifier: the classifier (can also be an Iterable of classifiers) to use at the end of the SCNN
        """
        self.classifier = classifier

    def predict_embeddings(self, x: np.ndarray):
        x = x.reshape(self.input_shape)
        accel_output = np.array(list(map(lambda s: self.model.execute(s).flatten(), x)))
        return accel_output.squeeze()

    def fit(self, x, y):
        """
        Fit the output classifier on the given data.

        Args:
            x: numpy data that is passed through the CNN before fitting
            y: labels
        """
        embeddings = self.predict_embeddings(x)
        self.classifier.fit(embeddings, y)

    def predict_proba(self, x):
        embeddings = self.predict_embeddings(x)
        return self.classifier.predict_proba(embeddings)

    def predict(self, x):
        embeddings = self.predict_embeddings(x)
        return self.classifier.predict(embeddings)


def pop_all_samples_from_redis(redis_inst: emager_redis.EmagerRedis):
    calib_to_pop = redis_inst.r.llen(redis_inst.LABELS_FIFO_KEY)
    calib_data, calib_labels = [], []
    for _ in range(calib_to_pop):
        new_data = redis_inst.pop_sample(True)
        calib_data.append(new_data[0])
        calib_labels.append(new_data[1])
    calib_data = np.array(calib_data)
    calib_labels = np.array(calib_labels)

    to_pop = redis_inst.r.llen(redis_inst.SAMPLES_FIFO_KEY)
    # print("Going to pop ", to_pop, " batches from redis")
    data = []
    for _ in range(to_pop):
        new_data = redis_inst.pop_sample()[0]
        data.append(new_data)
        # print(new_data.shape, len(new_data), len(data))
    data = np.array(data)
    return data.reshape(-1, 64), calib_data.reshape(-1, 64), calib_labels


def process_data(redis_inst: emager_redis.EmagerRedis, data: np.ndarray):
    transform = transforms.transforms_lut[redis_inst.get_str(redis_inst.TRANSFORM_KEY)]
    # linput_shape = (-1, *driver.io_shape_dict["ishape_normal"][0])
    return transform(data)


def push_preds_to_redis(redis_inst: emager_redis.EmagerRedis, preds: np.ndarray):
    redis_inst.r.lpush(
        redis_inst.PREDICTIONS_FIFO_KEY, preds.astype(np.uint8).tobytes()
    )


if __name__ == "__main__":
    import emager_py.data_generator as dg
    from emager_py.streamers import RedisStreamer
    from emager_py import emager_redis as er
    from emager_py import dataset as ed, data_processing as dp
    from sklearn.metrics import accuracy_score

    import utils
    import globals

    md = utils.get_model_params_from_disk()
    hostname = er.get_docker_redis_ip()

    # Load test data
    test_session = 2 if md["session"] == 1 else 1

    # Push data as fast as possible
    rs = RedisStreamer(hostname, False)
    rs.clear()
    rs.r.set_pynq_params(md["transform"])
    calib_data, test_data = ed.get_lnocv_datasets(
        globals.EMAGER_DATASET_ROOT, md["subject"], test_session, md["repetition"]
    )
    calib_data, calib_labels = dp.extract_labels(calib_data)
    test_data, test_labels = dp.extract_labels(test_data)

    # Take some calibration data
    for i in np.unique(calib_labels):
        matches = np.argwhere(calib_labels == i).flatten()
        idxs = np.random.choice(
            matches, 20 * globals.EMAGER_SAMPLE_BATCH, replace=False
        )
        for idx in idxs:
            rs.r.push_sample(calib_data[idx], calib_labels[idx])

    r = er.EmagerRedis(hostname)
    list(map(lambda x: rs.r.push_sample(x), test_data))

    classi = CosineSimilarity()
    accel = DummyFinnAccel()
    model = EmgSCNNWrapper(accel, classi, (1, 4, 16, 1))

    data, calib_data, calib_labels = pop_all_samples_from_redis(r)
    calib_data = process_data(r, calib_data)
    data = process_data(r, data)

    calib_labels = calib_labels[:: len(calib_labels) // len(calib_data)]
    test_labels = test_labels[:: len(test_labels) // len(data)]

    model.fit(calib_data, calib_labels)
    preds = model.predict(data)
    push_preds_to_redis(r, preds)

    print("Accuracy: ", accuracy_score(test_labels, preds, normalize=True) * 100)
