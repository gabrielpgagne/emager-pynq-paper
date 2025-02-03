import os
from typing import Iterable
from matplotlib.pylab import f
import torch
import json
import enum
import pandas as pd
import numpy as np
from scipy import signal

import emager_py.dataset as ed

from globals import OUT_DIR_ROOT, OUT_DIR_MODELS, OUT_DIR_STATS, OUT_DIR_FINN


def filter_notch(data: np.ndarray) -> np.ndarray:
    """Filter GRN{HW, C}-shaped data.

    Args:
        data (np.ndarray): EMG data to filter (N, 64) or (G, R, N, 64)

    Returns:
        _type_: _description_
    """
    b, a = signal.iirnotch(60, 30, 1000)
    if len(data.shape) == 2:
        # (N, 64)
        filtered = signal.lfilter(b, a, data, axis=0)
    elif len(data.shape) == 4:
        # (G, R, N, 64)
        filtered = signal.lfilter(b, a, data, axis=3)
    return filtered


def noise_floor(emg) -> float:
    """
    Calculate noise floor of CRN-shaped EMG data.
    """
    noise_floor = np.sqrt(np.mean((emg[3] - np.mean(emg[3])) ** 2))
    return noise_floor


class ModelMetric(enum.Enum):
    ACC_RAW_INTRA = "acc_raw_intra"
    ACC_MAJ_INTRA = "acc_maj_intra"
    ACC_RAW_INTER = "acc_raw_inter"
    ACC_MAJ_INTER = "acc_maj_inter"

    def __str__(self):
        return self.value


def parse_model_name(model_name: str):
    """
    Returns the session, cross-validation repetition, and quantization bits from a model name.

    The extension is ignored, as long as the project's formatting is enforced.

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


def format_model_root(subject, subdir, ensure_exists=False):
    """
    Format the root directory for a given subject.

    Params:
        - subject: the subject to consider
        - subdir: the base path, eg `globals.OUT_DIR_MODELS`, which is appended after `globals.OUT_DIR_ROOT`
        - ensure_exists: whether to create the directory if it does not exist

    Returns the formatted path.
    """
    path = OUT_DIR_ROOT + subdir + ed.format_subject(subject)
    if ensure_exists and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def format_model_name(
    out_dir,
    subject,
    session,
    cross_validation_rep,
    quant_bits,
    ensure_exists=False,
):
    """Format the model name for a given subject, session, cross-validation repetition, and quantization bits.

    Args:
        output_base: root output folder eg globals.OUT_DIR_STATS or globals.OUT_DIR_MODEL
        subject (str|int)
        session (str|int)
        cross_validation_rep (str|int): Cross validation repetition
        quant_bits (str|int): Number of quantization bits
        extension (str, optional): File extension. Defaults to ".pth".

    Returns:
        str: The full path of the formatted model name
    """
    extension = ".csv"
    if out_dir == OUT_DIR_MODELS:
        extension = ".pth"

    out_dir = format_model_root(subject, out_dir, ensure_exists=ensure_exists)

    if isinstance(session, str):
        session = int(session)
    if isinstance(quant_bits, str):
        quant_bits = int(quant_bits)
    if not isinstance(cross_validation_rep, Iterable):
        cross_validation_rep = [cross_validation_rep]

    cross_validation_rep = [f"{int(r):02d}" for r in cross_validation_rep]
    cross_validation_rep = "-".join(cross_validation_rep)

    if quant_bits < 0:
        quant_bits = 32

    out_path = (
        out_dir + f"s{session:01d}_cv{cross_validation_rep}_q{quant_bits}{extension}"
    )
    return out_path


def format_finn_output_dir(subject, quant_bits, shots):
    """
    Format the output directory for the FINN build flow, with the structure `output/finn/<subject>/<nq>quant_<ns>shots/`
    """
    build_dir = (
        format_model_root(subject, OUT_DIR_FINN, True)
        + f"{quant_bits:02d}quant_{shots:02d}shots/"
    )
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)
    return build_dir


def save_model(
    model: torch.nn.Module,
    metadata: pd.DataFrame,
    subject,
    session,
    cross_validation_rep,
    quant_bits,
):
    """
    Metadata must be the pytorch model's metadata. Not FINN evaluation results.
    """
    model_path = format_model_name(
        OUT_DIR_MODELS,
        subject,
        session,
        cross_validation_rep,
        quant_bits,
        ensure_exists=True,
    )
    metadata_path = format_model_name(
        OUT_DIR_STATS,
        subject,
        session,
        cross_validation_rep,
        quant_bits,
        ensure_exists=True,
    )
    model = model.to("cpu")
    torch.save(model.state_dict(), model_path)
    metadata.to_csv(metadata_path, index=False)

    return model_path, metadata_path


def load_metadata(
    base_path,
    subject,
    session,
    cross_validation_rep,
    quant_bits,
) -> pd.DataFrame:
    metadata_path = format_model_name(
        base_path, subject, session, cross_validation_rep, quant_bits
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
        OUT_DIR_MODELS,
        subject,
        session,
        cross_validation_rep,
        quant_bits,
        ensure_exists=False,
    )
    model = model.to("cpu")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model


def load_both(model, subject, session, cross_validation_rep, quant_bits):
    return load_model(
        model, subject, session, cross_validation_rep, quant_bits
    ), load_metadata(OUT_DIR_STATS, subject, session, cross_validation_rep, quant_bits)


def concat_metadata(base_dir, subject, quant_bits):
    """
    For a given subject and quantization bits, concatenate all metadata cross-validation files into a single DataFrame.

    Args:
        subject (int): The subject to consider
        quant_bits (int): The number of quantization bits to consider

    Returns:
        pd.DataFrame: The concatenated metadata with columns (shots, session, validation_rep, acc_raw, acc_maj).
        `None` if no metadata files exist.
    """
    if not isinstance(quant_bits, int):
        quant_bits = int(quant_bits)

    if quant_bits < 0:
        quant_bits = 32

    # For flexibility, list all files in the directory and filter by the quantization bits
    files = sorted(os.listdir(format_model_root(subject, base_dir)))

    big_metadata = None
    for f in files:
        if not f.endswith(".csv"):
            continue

        ses, rep, quant = parse_model_name(f)
        if quant != quant_bits:
            continue

        data = load_metadata(base_dir, subject, ses, rep, quant_bits)
        data.insert(1, "validation_rep", [rep] * len(data))
        data.insert(1, "session", [ses] * len(data))
        if big_metadata is None:
            big_metadata = data
            continue
        big_metadata = pd.concat([big_metadata, data], axis=0, ignore_index=True)

    return big_metadata


def get_average_across_validations(base_dir, subject: int, quant_bits: int):
    """
    Get the average accuracy across all cross-validation repetitions for a given subject and quantization bits.

    Params:
        - subject: the subject to consider
        - quant_bits: the number of quantization bits to consider

    Returns a DataFrame with the average accuracy across all cross-validation repetitions with keys (shots, acc_raw, acc_maj)
    """
    metadata = concat_metadata(base_dir, subject, quant_bits)
    # if metadata is None:
    #   print("Could not fetch metadata across validations")
    #   return pd.DataFrame({"shots": [], ModelMetric.ACC_RAW: [], ModelMetric.ACC_MAJ: []})
    metadata.pop("session")
    metadata.pop("validation_rep")
    return metadata.groupby("shots", as_index=False).mean()


def get_all_accuracy_vs_shots(base_dir, quant_bits: int):
    """
    Get the average accuracy across all subjects for a given quantization bits.

    Params:
        - quant_bits: the number of quantization bits to consider

    Returns a list of DataFrame with the average accuracy across all subjects with keys (shots, acc_raw, acc_maj)
    """
    models_root = OUT_DIR_ROOT + base_dir
    subjects = list(
        filter(lambda f: os.path.isdir(models_root + f), os.listdir(models_root))
    )
    subjects.sort()
    metadata = []
    for subject in subjects:
        md = get_average_across_validations(base_dir, subject, quant_bits)
        metadata.append(md)
        # if metadata is None:
        #    metadata = md
        #    continue
        # metadata = pd.concat([metadata, md], axis=0)
    return metadata


def get_all_accuracy_vs_quant(base_dir: str, n_shots: int):
    """Get the average accuracy across all subjects for a given number of shots.

    Args:
        n_shots (int): Number of shots

    Returns:
        list[DataFrame]: A list of DataFrames with the average accuracy across all subjects with keys (quantization, acc_raw, acc_maj)
    """
    models_root = OUT_DIR_ROOT + base_dir
    subjects = list(
        filter(lambda f: os.path.isdir(models_root + f), os.listdir(models_root))
    )
    subjects.sort()
    all_metadata = []
    for subject in subjects:
        metadata = None
        files = list(
            filter(lambda f: f.endswith(".csv"), os.listdir(models_root + subject))
        )
        for f in files:
            _, _, quant_bits = parse_model_name(f)
            if quant_bits < 0:
                quant_bits = 32
            md = get_average_across_validations(base_dir, subject, quant_bits)
            mask = md["shots"].isin([n_shots])
            md = md[mask]
            md.pop("shots")
            md.insert(0, "quantization", [quant_bits] * len(md))
            if metadata is None:
                metadata = md
                continue
            metadata = pd.concat([metadata, md], axis=0, ignore_index=True)
        all_metadata.append(metadata.groupby("quantization", as_index=False).mean())
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

    results = concat_metadata(OUT_DIR_STATS, subject, quant)
    mask = results["shots"].isin([n_shots])
    results = results[mask]
    best = results.loc[results[metric.value].idxmax()]
    return (
        int(best["session"]),
        [int(r) for r in best["validation_rep"]],
        best[metric.value],
    )


def get_model_params_from_disk() -> dict:
    """
    Load the current model to build dict from `OUT_DIR_ROOT + OUT_DIR_FINN + "finn_config.json"`
    """
    dic = dict()
    with open(OUT_DIR_ROOT + OUT_DIR_FINN + "finn_config.json", "r") as f:
        dic = json.load(f)
    return dic


def get_accelerator_resources(
    subject,
    quant,
    shots,
) -> dict:
    """
    Load the accelerator resources from `OUT_DIR_ROOT + OUT_DIR_FINN + "resources.json"`
    """
    subject = ed.format_subject(subject)
    quant = f"{quant:02d}"
    shots = f"{shots:02d}"

    val = 0

    with open(
        OUT_DIR_ROOT
        + OUT_DIR_FINN
        + f"{subject}/{quant}quant_{shots}shots/report/post_synth_resources.xml",
        "r",
    ) as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if "top_StreamingDataflowPartition_1_0" in line:
                lut_line = lines[i + 1]
                idx = lut_line.find("contents=")
                idx2 = lut_line.find("halign=") - 2
                val = int(lut_line[idx + 10 : idx2])
                break
    return val


def lock_finn(**kwargs):
    with open(f"{OUT_DIR_ROOT}/{OUT_DIR_FINN}/lock.txt", "w") as f:
        f.write(f"locked by {kwargs}")


def unlock_finn():
    os.remove(f"{OUT_DIR_ROOT}/{OUT_DIR_FINN}/lock.txt")


def is_finn_locked():
    if os.path.exists(f"{OUT_DIR_ROOT}/{OUT_DIR_FINN}/lock.txt"):
        return True
    else:
        return False


def resume_from_latest(cross_validations: list, quantizations: list[int]):
    """
    Returns subject, session, cross-validation, and quantization to resume training from the latest model.
    """
    cross_validations = [[int(v) for v in cv] for cv in cross_validations]
    sub0, ses0, cv0, q0 = 0, 0, 0, 0

    # Get all subjects
    try:
        subjs = ed.get_subjects(OUT_DIR_ROOT + OUT_DIR_MODELS)
        if len(subjs) == 0:
            return sub0, ses0, cv0, q0
        sub0 = len(subjs) - 1
    except FileNotFoundError:
        return sub0, ses0, cv0, q0

    # Get all the models for the latest subject
    models = os.listdir(OUT_DIR_ROOT + OUT_DIR_MODELS + subjs[-1])
    if len(models) == 0:
        return sub0, ses0, cv0, q0

    ses0 = int(max([m.split("_")[0][1:] for m in models])) - 1  # session 1-indexed
    models = list(filter(lambda e: e[1:].startswith(str(ses0 + 1)), models))
    if len(models) == 0:
        return sub0, ses0, cv0, q0

    # now find max cross-validation and filter models
    for model in models:
        _, cv, _ = parse_model_name(model)
        where = cross_validations.index(cv)
        if where > cv0:
            cv0 = where
    models = list(
        filter(
            lambda model: parse_model_name(model)[1] == cross_validations[cv0],
            models,
        )
    )
    if len(models) == 0:
        return sub0, ses0, cv0, q0

    # finally find max quantization and filter models
    for model in models:
        _, _, q = parse_model_name(model)
        where = quantizations.index(q)
        if where > q0:
            q0 = where

    lims = [
        len(quantizations),
        len(cross_validations),
        len(ed.get_sessions()),
        len(subjs),
    ]
    resume = [q0, cv0, ses0, sub0]
    for i, (res, lim) in enumerate(list(zip(resume, lims))[:-1]):
        r0 = (res + 1) % lim
        resume[i] = r0
        if not r0 < res:
            # no rollover so break
            break
        # rollover
        resume[i + 1] += 1
    return tuple(reversed(resume))


if __name__ == "__main__":
    import emager_py.torch.models as etm

    # print(format_model_root(5))
    # test = format_model_name(1, 1, [2, 3], 3)
    # print(test)
    # print(parse_model_name(test))

    # print(save_model(torch.nn.Linear(5, 5), pd.DataFrame({"test"}), 1, 2, [9, 1], 2))
    # print(load_metadata(1, 2, [9, 1], 2))

    # ret = concat_metadata(0, 1)
    # print(ret)
    # print("*" * 80)

    # ret = get_average_across_validations(0, 4)
    # print(ret)
    # print("*" * 80)

    for subject in [0, 1]:
        for quant in [2, 3, 4, 6, 8]:
            print(
                f"Total LUTs for {subject} @ {quant}-bits:",
                get_accelerator_resources(subject, quant, 10),
            )
    ret = get_best_model(0, 3, 10, ModelMetric.ACC_MAJ)
    load_model(etm.EmagerSCNN((4, 16), 3), 0, ret[0], ret[1], 3)

    print(ret)
    print("*" * 80)

    print(get_all_accuracy_vs_quant(-1)[0])
    print(get_all_accuracy_vs_shots(3)[0])
    # mask = ret["shots"].isin([-1])
    # print(ret[mask])
