"""
This module is used to train all subjects with the SCNN model and do preliminary testing.
"""

from typing import Iterable
import pandas as pd
from datetime import datetime

import numpy as np
import torch.cuda
from torch.utils.data import DataLoader
import lightning as L

from sklearn.metrics import accuracy_score

import emager_py.dataset as ed
import emager_py.data_processing as dp
import emager_py.torch.models as etm
import emager_py.transforms as etrans
import emager_py.torch.datasets as etd
import emager_py.torch.utils as etu
import emager_py.majority_vote as emv

from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from siamese_classifier import CosineSimilarity

import utils
import globals as g


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


def train_all_scnn(cross_validations: list[str], quantizations: list[int], transform):
    from multiprocessing import Process
    import time

    num_procs = 4

    if not isinstance(quantizations, list):
        quantizations = [quantizations]

    sessions = ed.get_sessions()

    first_run = True
    sub0, ses0, cv0, q0 = utils.resume_from_latest(cross_validations, quantizations)

    for subj in ed.get_subjects(g.EMAGER_DATASET_ROOT)[sub0:]:
        ses_start = ses0 if first_run else 0
        for ses in sessions[ses_start:]:
            cv_start = cv0 if first_run else 0
            for valid_reps in cross_validations[cv_start:]:
                q_start = q0 if first_run else 0
                first_run = False

                procs = []
                while len(quantizations[q_start:]) % num_procs != 0:
                    quantizations.append(None)

                for quants in zip(
                    quantizations[q_start::2], quantizations[q_start + 1 :: 2]
                ):
                    print("*" * 100)
                    print(f"Current datetime: {datetime.now()}")
                    print(
                        f"Training subject {subj} on session {ses} with L{len(valid_reps)}OCV reps={valid_reps} with {quants}-bit quantization."
                    )
                    print("*" * 100)

                    def _p(quant):
                        if quant is None:
                            return

                        model, test_results = train_scnn(
                            g.EMAGER_DATASET_ROOT,
                            subj,
                            ses,
                            valid_reps,
                            transform,
                            quant,
                        )
                        utils.save_model(
                            model, test_results, subj, ses, valid_reps, quant
                        )

                    for q in quants:
                        p = Process(target=_p, args=(q,))
                        p.start()
                        procs.append(p)
                        time.sleep(10)

                    for p in procs:
                        p.join()


def test_all_scnn(cross_validations: list[str], quantizations: list[int], transform):
    if not isinstance(quantizations, list):
        quantizations = [quantizations]

    for subj in ed.get_subjects(g.EMAGER_DATASET_ROOT):
        for ses in ed.get_sessions():
            for valid_reps in cross_validations:
                for quant in quantizations:
                    print("*" * 100)
                    print(f"Current datetime: {datetime.now()}")
                    print(
                        f"Training subject {subj} on session {ses} with L{len(valid_reps)}OCV reps={valid_reps} with {quant}-bit quantization."
                    )
                    print("*" * 100)

                    _, calib_intra, test_intra, calib_inter, test_inter = (
                        etd.get_triplet_dataloaders(
                            g.EMAGER_DATASET_ROOT,
                            subj,
                            ses,
                            valid_reps,
                            transform=transform,
                        )
                    )
                    model = etm.EmagerSCNN(quant)
                    model = utils.load_model(model, subj, ses, valid_reps, quant)
                    intra_test_results = test_scnn(
                        model, test_intra, calib_intra, transform
                    )
                    inter_test_results = test_scnn(
                        model, test_inter, calib_inter, transform
                    )
                    test_results = pd.DataFrame(
                        {
                            "shots": intra_test_results["shots"],
                            "acc_raw_intra": intra_test_results["acc_raw"],
                            "acc_maj_intra": intra_test_results["acc_maj"],
                            "acc_raw_inter": inter_test_results["acc_raw"],
                            "acc_maj_inter": inter_test_results["acc_maj"],
                        }
                    )
                    utils.save_model(model, test_results, subj, ses, valid_reps, quant)


if __name__ == "__main__":
    L.seed_everything(310)
    torch.set_float32_matmul_precision("high")

    # ============ Train all models ==========

    # cross_validations = list(zip(ed.get_repetitions()[::2], ed.get_repetitions()[1::2]))
    # quantizations = [1, 2, 3, 4, 6, 8, 32]

    # train_all_scnn(cross_validations, quantizations, etrans.root_processing)

    # test_all_scnn(cross_validations, quantizations, etrans.root_processing)

    # ============ Single model parameters ==========

    SUBJECT = 13
    SESSION = 2
    VALID_REPS = [0]
    QUANT = 4

    # ========= Train a single model ==========

    model, results = train_scnn(
        g.EMAGER_DATASET_ROOT,
        SUBJECT,
        1,
        VALID_REPS,
        etrans.root_processing,
        QUANT,
        [-1],
    )
    utils.save_model(model, results, SUBJECT, SESSION, VALID_REPS, QUANT)
    print(results)

    # ========= Test a single model ==========

    # _, calib_intra, test_intra, calib_inter, test_inter = etd.get_triplet_dataloaders(
    #     g.EMAGER_DATASET_ROOT,
    #     SUBJECT,
    #     SESSION,
    #     VALID_REPS,
    #     transform=etrans.root_processing,
    # )
    # model = etm.EmagerSCNN(QUANT)
    # model = utils.load_model(model, SUBJECT, SESSION, VALID_REPS, QUANT)
    # intra_test_results = test_scnn(
    #     model, test_intra, calib_intra, etrans.root_processing, [-1]
    # )
    # inter_test_results = test_scnn(
    #     model, test_inter, calib_inter, etrans.root_processing, [-1]
    # )
    # results = pd.DataFrame(
    #     {
    #         "shots": intra_test_results["shots"],
    #         "acc_raw_intra": intra_test_results["acc_raw"],
    #         "acc_maj_intra": intra_test_results["acc_maj"],
    #         "acc_raw_inter": inter_test_results["acc_raw"],
    #         "acc_maj_inter": inter_test_results["acc_maj"],
    #     }
    # )
    # print(results)
