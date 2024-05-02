import os
import torch
import json
import enum
import numpy as np
import pandas as pd

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

    Params:
        - model_name: the name of the model file, e.g. "s1_cv01-02_q32.pth", can also be a full path.

    For unquantized, the filename ends in q32
    """
    model_name = model_name.split("/")[-1].split(".")[0]
    parts = model_name.split("_")
    session = int(parts[0][1:])
    cross_validation_rep = [int(p) for p in parts[1].replace("cv", "").split("-")]
    quant_bits = int(parts[2][1:])
    return session, cross_validation_rep, quant_bits


def format_model_root(subject, metadata=False, ensure_exists=False):
    """
    Format the root directory for a given subject.

    If metadata is True, the directory will be formatted for metadata files.

    If ensure_exists is True, the directory will be created if it does not exist.

    Returns the formatted path.
    """
    path = ""
    if metadata:
        path = OUT_DIR_STATS + ed.format_subject(subject)
    else:
        path = OUT_DIR_MODELS + ed.format_subject(subject)

    if ensure_exists and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    return path


def format_model_name(
    subject,
    session,
    cross_validation_rep,
    quant_bits,
    extension=".pth",
    ensure_exists=False,
):
    """Format the model name for a given subject, session, cross-validation repetition, and quantization bits.

    Args:
        subject (str|int)
        session (str|int)
        cross_validation_rep (str|int): Cross validation repetition
        quant_bits (str|int): Number of quantization bits
        extension (str, optional): File extension. Defaults to ".pth".

    Returns:
        str: The formatted model name
    """
    if not extension.startswith("."):
        extension = "." + extension

    out_dir = ""
    if extension != ".pth":
        out_dir = format_model_root(subject, True, ensure_exists=ensure_exists)
    else:
        out_dir = format_model_root(subject, False, ensure_exists=ensure_exists)

    if isinstance(session, str):
        session = int(session)
    if isinstance(quant_bits, str):
        quant_bits = int(quant_bits)
    if not isinstance(cross_validation_rep, list):
        cross_validation_rep = [cross_validation_rep]

    cross_validation_rep = [f"{int(r):02d}" for r in cross_validation_rep]
    cross_validation_rep = "-".join(cross_validation_rep)

    if quant_bits < 0:
        quant_bits = 32

    out_path = (
        out_dir + f"s{session:01d}_cv{cross_validation_rep}_q{quant_bits}{extension}"
    )
    return out_path


def save_model(
    model: torch.nn.Module,
    metadata: pd.DataFrame,
    subject,
    session,
    cross_validation_rep,
    quant_bits,
):
    model_path = format_model_name(
        subject,
        session,
        cross_validation_rep,
        quant_bits,
        extension=".pth",
        ensure_exists=True,
    )
    metadata_path = format_model_name(
        subject,
        session,
        cross_validation_rep,
        quant_bits,
        extension=".csv",
        ensure_exists=True,
    )

    torch.save(model.state_dict(), model_path)
    metadata.to_csv(metadata_path, index=False)

    return model_path, metadata_path


def load_metadata(
    subject,
    session,
    cross_validation_rep,
    quant_bits,
) -> pd.DataFrame:
    metadata_path = format_model_name(
        subject, session, cross_validation_rep, quant_bits, extension=".csv"
    )
    return pd.read_csv(metadata_path)


def load_model(
    model: torch.nn.Module,
    subject,
    session,
    cross_validation_rep,
    quant_bits,
):
    model_path = format_model_name(
        subject,
        session,
        cross_validation_rep,
        quant_bits,
        extension=".pth",
        ensure_exists=False,
    )

    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model


def load_both(model, subject, session, cross_validation_rep, quant_bits):
    return load_model(
        model, subject, session, cross_validation_rep, quant_bits
    ), load_metadata(subject, session, cross_validation_rep, quant_bits)


def concat_metadata(subject, quant_bits):
    """For a given subject and quantization bits, concatenate all metadata cross-validation files into a single DataFrame.

    Args:
        subject (int): The subject to consider
        quant_bits (int): The number of quantization bits to consider

    Returns:
        pd.DataFrame: The concatenated metadata with columns (shots, session, validation_rep, acc_raw, acc_maj).
    """
    if not isinstance(quant_bits, int):
        quant_bits = int(quant_bits)

    # For flexibility, list all files in the directory and filter by the quantization bits
    files = sorted(os.listdir(OUT_DIR_STATS + ed.format_subject(subject)))

    big_metadata = None
    for f in files:
        ses, rep, quant = parse_model_name(f)
        if quant != quant_bits:
            continue

        data = load_metadata(subject, ses, rep, quant_bits)
        data.insert(1, "validation_rep", [rep] * len(data))
        data.insert(1, "session", [ses] * len(data))
        if big_metadata is None:
            big_metadata = data
            continue
        big_metadata = pd.concat([big_metadata, data], axis=0, ignore_index=True)

    return big_metadata


def get_average_across_validations(subject: int, quant_bits: int):
    """
    Get the average accuracy across all cross-validation repetitions for a given subject and quantization bits.

    Params:
        - subject: the subject to consider
        - quant_bits: the number of quantization bits to consider

    Returns a DataFrame with the average accuracy across all cross-validation repetitions with keys (shots, acc_raw, acc_maj)
    """
    if not isinstance(quant_bits, int):
        quant_bits = int(quant_bits)

    metadata = concat_metadata(subject, quant_bits)
    metadata.pop("session")
    metadata.pop("validation_rep")
    return metadata.groupby("shots", as_index=False).mean()


def get_all_accuracy_vs_shots(quant_bits: int):
    """
    Get the average accuracy across all subjects for a given quantization bits.

    Params:
        - quant_bits: the number of quantization bits to consider

    Returns a list of DataFrame with the average accuracy across all subjects with keys (shots, acc_raw, acc_maj)
    """
    if not isinstance(quant_bits, int):
        quant_bits = int(quant_bits)
    subjects = os.listdir(OUT_DIR_STATS)
    metadata = []
    for subject in subjects:
        md = get_average_across_validations(subject, quant_bits)
        metadata.append(md)
        # if metadata is None:
        #    metadata = md
        #    continue
        # metadata = pd.concat([metadata, md], axis=0)
    assert len(metadata) == len(subjects)
    return metadata


def get_all_accuracy_vs_quant(n_shots: int):
    """Get the average accuracy across all subjects for a given number of shots.

    Args:
        n_shots (int): Number of shots

    Returns:
        list[DataFrame]: A list of DataFrames with the average accuracy across all subjects with keys (quantization, acc_raw, acc_maj)
    """
    subjects = os.listdir(OUT_DIR_STATS)
    all_metadata = []
    for subject in subjects:
        metadata = None
        for f in os.listdir(OUT_DIR_STATS + subject):
            if not f.endswith(".csv"):
                continue
            _, _, quant_bits = parse_model_name(f)
            if quant_bits < 0:
                quant_bits = 32
            md = get_average_across_validations(subject, quant_bits)
            mask = md["shots"].isin([n_shots])
            md = md[mask]
            md.pop("shots")
            md.insert(0, "quantization", [quant_bits] * len(md))
            if metadata is None:
                metadata = md
                continue
            metadata = pd.concat([metadata, md], axis=0, ignore_index=True)
        all_metadata.append(metadata.groupby("quantization", as_index=False).mean())
    assert len(all_metadata) == len(subjects)
    return all_metadata


def get_best_model(subject, quant, n_shots, metric: ModelMetric):
    """
    Get the best model for a given subject and quantization bits.

    Params:
        - quantized_subject_stats: the output of get_accuracy_statistics
        - n_shots: the number of shots to consider, -1 meaning "all shots"

    Returns (best_session, best_validation_rep, best_accuracy)
    """
    if isinstance(n_shots, str):
        n_shots = int(n_shots)

    results = concat_metadata(subject, quant)
    mask = results["shots"].isin([n_shots])
    results = results[mask]
    best = results.loc[results[metric.value].idxmax()]
    return (
        int(best["session"]),
        [int(r) for r in best["validation_rep"]],
        best[metric.value],
    )


if __name__ == "__main__":
    import emager_py.torch.models as etm

    # print(format_model_root(5))
    # test = format_model_name(1, 1, [2, 3], 3)
    # print(test)
    # print(parse_model_name(test))

    # print(save_model(torch.nn.Linear(5, 5), pd.DataFrame({"test"}), 1, 2, [9, 1], 2))
    # print(load_metadata(1, 2, [9, 1], 2))

    ret = concat_metadata(0, 1)
    print(ret)
    print("*" * 80)

    ret = get_average_across_validations(0, 4)
    print(ret)
    print("*" * 80)

    ret = get_best_model(0, 4, -1, ModelMetric.ACC_RAW)
    load_model(etm.EmagerSCNN((4, 16), 4), 0, ret[0], ret[1], 4)

    print(ret)
    print("*" * 80)

    print(get_all_accuracy_vs_quant(-1)[0])
    print(get_all_accuracy_vs_shots(4)[0])
    # mask = ret["shots"].isin([-1])
    # print(ret[mask])
