from typing import Iterable
import pandas as pd

import numpy as np
import torch.cuda
from torch.utils.data import DataLoader
import lightning as L

from sklearn.metrics import accuracy_score

import emager_py.torch.models as etm
import emager_py.transforms as etrans
import emager_py.torch.datasets as etd
import emager_py.torch.utils as etu
import emager_py.majority_vote as emv

from siamese_classifier import CosineSimilarity

import utils
import globals as g


def train_scnn(
    data_root,
    subject,
    train_session,
    valid_reps,
    transform,
    quant,
    shots=[1, 2, 3, 5, 7, 10, 13, 15, 18, 20, -1],
):
    if isinstance(transform, str):
        transform = etrans.transforms_lut[g.TRANSFORM]

    if not isinstance(shots, Iterable):
        shots = list(shots)

    # Boilerplate
    trainer = L.Trainer(
        accelerator="auto" if torch.cuda.is_available() or quant == -1 else "cpu",
        enable_checkpointing=False,
        logger=False,
        max_epochs=10,
    )
    train, calib_intra, test_intra, calib_inter, test_inter = (
        etd.get_triplet_dataloaders(
            data_root, subject, train_session, valid_reps, transform=transform
        )
    )

    # Train and test
    model = etm.EmagerSCNN(quant)
    trainer.fit(model, train)
    intra_test_results = test_scnn(model, test_intra, calib_intra, transform, shots)
    inter_test_results = test_scnn(model, test_inter, calib_inter, transform, shots)
    test_results = pd.DataFrame(
        {
            "shots": intra_test_results["shots"],
            "acc_raw_intra": intra_test_results["acc_raw"],
            "acc_maj_intra": intra_test_results["acc_maj"],
            "acc_raw_inter": inter_test_results["acc_raw"],
            "acc_maj_inter": inter_test_results["acc_maj"],
        }
    )
    return model, test_results


def test_scnn(
    model: etm.EmagerSCNN,
    test_dataloader: DataLoader,
    calib_dataloader: DataLoader,
    transform: str,
    shots=[1, 2, 3, 5, 7, 10, 13, 15, 18, 20, -1],
    n_trials=5,
):
    test_dict = {
        "shots": [],
        "acc_raw": [],
        "acc_maj": [],
    }

    n_votes = 150 // etrans.get_transform_decimation(transform)

    # Calibration embeddings
    calib_embeddings, calib_labels = etu.get_all_embeddings(
        model, calib_dataloader, model.device
    )
    test_embeddings, test_labels = etu.get_all_embeddings(
        model, test_dataloader, model.device
    )

    test_labels_mv = emv.majority_vote(test_labels, n_votes)

    for shot in shots:
        test_dict["shots"].append(shot)
        tmp_dict = {
            "acc_raw": [],
            "acc_maj": [],
        }
        for _ in range(n_trials):
            # n_shots_embeddings = dp.get_n_shot_embeddings(embeddings, labels, 6, shot)

            # First, calibrate the classifier with fit()
            # so we must get n-shot embeddings and the labels
            calib_embeds_trial, calib_labels_trial = calib_embeddings, calib_labels
            if shot != -1:
                to_sample = np.zeros((0,), dtype=np.uint8)
                for k in np.unique(calib_labels):
                    num_k = np.sum([calib_labels == k])
                    to_sample_k = np.random.choice(
                        np.where(calib_labels == k)[0],
                        min(shot, num_k),
                        replace=False,
                    )
                    to_sample = np.append(to_sample, to_sample_k)

                calib_embeds_trial, calib_labels_trial = (
                    calib_embeddings[to_sample],
                    calib_labels[to_sample],
                )

            # Create classifier and calibrate it
            classi = CosineSimilarity()
            # classi = LinearDiscriminantAnalysis()
            # classi = KNeighborsClassifier()

            classi.fit(calib_embeds_trial, calib_labels_trial)

            # Get all predictions
            test_preds = classi.predict(test_embeddings)
            test_preds_mv = emv.majority_vote(test_preds, n_votes)

            # Now get accuracy results
            raw_acc = accuracy_score(test_labels, test_preds)
            majority_acc = accuracy_score(test_labels_mv, test_preds_mv)

            # Do dictionary stuff
            tmp_dict["acc_raw"].append(raw_acc)
            tmp_dict["acc_maj"].append(majority_acc)

            if shot == -1:
                # no point in reiterating
                break

        for k, v in tmp_dict.items():
            if k not in test_dict:
                test_dict[k] = []
            test_dict[k].append(sum(v) / len(v))

    return test_dict


if __name__ == "__main__":
    L.seed_everything(310)
    torch.set_float32_matmul_precision("high")

    # ============ Single model parameters ==========

    SUBJECT_TRAIN = 0
    SUBJECT_TEST = 1
    SESSION = 1
    VALID_REPS = [0, 1]
    QUANT = 8

    # ========= Train a single model ==========

    model, results = train_scnn(
        g.EMAGER_DATASET_ROOT,
        SUBJECT_TRAIN,
        SESSION,
        VALID_REPS,
        etrans.root_processing,
        QUANT,
        [-1],
    )
    utils.save_model(model, results, SUBJECT_TRAIN, SESSION, VALID_REPS, QUANT)
    print(results)

    # ========= Test a single model ==========

    _, _, _, calib_inter, test_inter = etd.get_triplet_dataloaders(
        g.EMAGER_DATASET_ROOT,
        SUBJECT_TEST,
        SESSION,
        VALID_REPS,
        transform=etrans.root_processing,
    )
    model = etm.EmagerSCNN(QUANT)
    model = utils.load_model(model, SUBJECT_TRAIN, SESSION, VALID_REPS, QUANT)
    inter_test_results = test_scnn(
        model, test_inter, calib_inter, etrans.root_processing, [-1]
    )
    results = pd.DataFrame(
        {
            "shots": inter_test_results["shots"],
            "acc_raw_inter": inter_test_results["acc_raw"],
            "acc_maj_inter": inter_test_results["acc_maj"],
        }
    )
    print(results)
