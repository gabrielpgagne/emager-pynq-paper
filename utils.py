import os
import torch
import json
import numpy as np
import enum

import emager_py.dataset as ed

from globals import OUT_DIR_MODELS, OUT_DIR_STATS


class ModelMetric(enum.Enum):
    ACC_RAW = "acc_raw"
    ACC_MAJ = "acc_maj"

    def __str__(self):
        return self.value


def parse_model_name(model_name: str):
    """
    Returns the session, cross-validation repetition, and quantization bits from a model name.

    For unquantized, the filename ends in q32
    """
    model_name = model_name.split("/")[-1].split(".")[0]
    parts = model_name.split("_")
    session = int(parts[0][1:])
    cross_validation_rep = int(parts[1][2:])
    quant_bits = int(parts[2][1:])
    return session, cross_validation_rep, quant_bits


def format_model_root(subject, metadata=False):
    if metadata:
        return OUT_DIR_STATS + ed.format_subject(subject) + "metadata/"
    return OUT_DIR_MODELS + ed.format_subject(subject)


def format_model_name(
    subject, session, cross_validation_rep, quant_bits, extension=".pth"
):
    if not extension.startswith("."):
        extension = "." + extension

    out_dir = format_model_root(subject)
    if extension != ".pth":
        out_dir = format_model_root(subject, True)

    if isinstance(session, str):
        session = int(session)
    if isinstance(cross_validation_rep, str):
        cross_validation_rep = int(cross_validation_rep)
    if isinstance(quant_bits, str):
        quant_bits = int(quant_bits)

    if quant_bits < 0:
        quant_bits = 32

    out_path = (
        out_dir
        + f"s{session:03d}_cv{cross_validation_rep:03d}_q{quant_bits}{extension}"
    )
    return out_path


def save_model(
    model: torch.nn.Module,
    metadata: dict,
    subject,
    session,
    cross_validation_rep,
    quant_bits,
):
    base_path = format_model_root(subject)

    if not os.path.exists(base_path):
        os.makedirs(os.path.dirname(base_path), exist_ok=True)

    model_path = format_model_name(
        subject, session, cross_validation_rep, quant_bits, extension=".pth"
    )
    metadata_path = format_model_name(
        subject, session, cross_validation_rep, quant_bits, extension=".json"
    )

    torch.save(model.state_dict(), model_path)
    json.dump(metadata, open(metadata_path, "w"))

    return model_path, metadata_path


def load_metadata(
    subject,
    session,
    cross_validation_rep,
    quant_bits,
) -> dict:
    metadata_path = format_model_name(
        subject, session, cross_validation_rep, quant_bits, extension=".json"
    )

    return json.load(open(metadata_path, "r"))


def load_model(
    model: torch.nn.Module,
    subject,
    session,
    cross_validation_rep,
    quant_bits,
):
    model_path = format_model_name(
        subject, session, cross_validation_rep, quant_bits, extension=".pth"
    )

    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model


def load_both(model, subject, session, cross_validation_rep, quant_bits):
    return load_model(
        model, subject, session, cross_validation_rep, quant_bits
    ), load_metadata(subject, session, cross_validation_rep, quant_bits)


def get_accuracy_statistics(
    subject: int, quant_bits: int, metric: ModelMetric = ModelMetric.ACC_RAW
):
    """
    Get the average accuracy for a given subject and quantization bits.

    Params:
        - subject: the subject to consider
        - quant_bits: the number of quantization bits to consider, 32 being unquantized
        - metric: the metric to consider, one of `./models/subject/xyz.json`'s keys, except "shots".

    Returns a dict with shape:
    {
        "nshot1": {
            "acc": [0.1, 0.2, 0.3, ...],
            "acc_avg": 0.2,
            "acc_std": 0.05
        }
        "nshot2": {
            ...
        }
    }

    Where:
        - "shots" is the number of shots, -1 meaning "all shots"
        - "acc" is a list of accuracies for each cross-validation repetition
        - "acc_avg" is the average accuracy across all repetitions
        - "acc_std" is the standard deviation of the accuracy across all repetitions

    Example:
    >>> get_accuracy_statistics(0, 32, "acc_maj") # load the majority vote accuracy for subject 0 with 32 quantization bits

    """

    assert isinstance(metric, ModelMetric)

    metric = str(metric)

    if not isinstance(quant_bits, int):
        quant_bits = int(quant_bits)

    ret = dict(cross_validations=[])

    # For flexibility, list all files in the directory and filter by the quantization bits
    files = sorted(
        filter(
            lambda x: x.endswith(".json"),
            os.listdir(OUT_DIR_MODELS + ed.format_subject(subject)),
        )
    )

    # print(files)
    for f in files:
        ses, rep, quant = parse_model_name(f)
        if quant != quant_bits:
            continue
        if f"cv{rep:03d}" not in ret:
            ret["cross_validations"].append(rep)
        data = load_metadata(subject, ses, rep, quant_bits)
        for k in data["shots"]:
            k = str(k)
            if k not in ret:
                ret[k] = dict()
            if "acc" not in ret[k]:
                ret[k]["acc"] = []
            kidx = data["shots"].index(int(k))
            ret[k]["acc"].append(data[metric][kidx])

    for k, v in ret.items():
        if not k.isnumeric() or "acc" not in v:
            continue
        acc = np.array(v["acc"])
        mean = np.mean(acc)
        std = np.std(acc)
        ret[k]["acc_avg"] = mean
        ret[k]["acc_std"] = std

    return ret


def get_best_model(quantized_subject_stats: dict, n_shots):
    """
    Get the best model for a given subject and quantization bits.

    Params:
        - quantized_subject_stats: the output of get_accuracy_statistics
        - n_shots: the number of shots to consider, -1 meaning "all shots"

    Returns the cross-validation repetition with the highest accuracy.
    """
    if isinstance(n_shots, int):
        n_shots = str(n_shots)

    results = quantized_subject_stats[n_shots]
    best_idx = np.argmax(results["acc"])
    best_model = quantized_subject_stats["cross_validations"][best_idx]
    return best_model


def get_acc_vs_quant(subject, metric: ModelMetric, shots=-1):
    """
    Get the accuracy for a given subject across all quantization bits.

    Params:
        - subject: the subject to consider
        - metric: the metric to consider
        - shots: the number of shots to consider, -1 meaning "all shots"

    Returns a dict with shape:
    {
        n1: 0.1,
        n2: 0.2,
        n3: 0.3,
        ...
    }

    Where the keys `n` are the quantization bits and the values are the average accuracy across all cross-validation repetitions.
    """
    shots = str(shots)
    quants = set()
    files = os.listdir(OUT_DIR_MODELS + ed.format_subject(subject))
    for f in files:
        if not f.endswith(".json"):
            continue
        _, _, quant = parse_model_name(f)

        if quant not in quants:
            quants.add(int(quant))

    ret = {}
    for q in quants:
        stats = get_accuracy_statistics(subject, q, metric)
        ret[q] = stats[shots]["acc_avg"]
    return ret


def get_all_statistics(quantization, metric) -> list[dict]:
    """
    Retrieve a list of statistics from all available subjects.

    Simply returns a list of return values of `get_accuracy_statistics`.
    """
    stats = []
    for subject in os.listdir(OUT_DIR_MODELS):
        stats.append(get_accuracy_statistics(subject, quantization, metric))
    return stats


if __name__ == "__main__":
    import emager_py.torch.models as etm

    # print(format_model_root(5))
    # test = format_model_name(1, 1, , 3)
    # print(parse_model_name(test))
    # print(save_model(torch.nn.Linear(5, 5), {"key0": [0, 1, 2, 3]}, 1, 2, 9, 2))

    # print(load_model(etm.EmagerSCNN((4, 16), 2), 0, 1, 0, 2))
    # print(load_metadata(0, 1, 0, 2))
    # print(load_both(etm.EmagerSCNN((4, 16), 3), 0, 1, 0, 3))

    ret = get_accuracy_statistics(0, 32, ModelMetric.ACC_RAW)
    print(ret)
    print(get_acc_vs_quant(0, ModelMetric.ACC_RAW))
    for k, v in ret.items():
        if not k.isnumeric():
            continue

        best = get_best_model(ret, -1)
        print(f"{k}-shot best model: {best}")
