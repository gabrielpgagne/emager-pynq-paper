"""
This module is used to train all subjects with the SCNN model and do preliminary testing.
"""

import pandas as pd
from datetime import datetime

import numpy as np

import torch.cuda
from torch import nn
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import emager_py.dataset as ed
import emager_py.torch.models as etm
import emager_py.transforms as etrans
import emager_py.torch.datasets as etd
import emager_py.majority_vote as emv

import globals as g
import utils


def test_cnn(
    model: etm.EmagerCNN,
    trainer: L.Trainer,
    test_dataloader: DataLoader,
    calib_dataloader: None | DataLoader,
    transform: str,
):
    n_votes = 150 // etrans.get_transform_decimation(transform)

    if calib_dataloader is not None:
        model.set_finetune()
        trainer.fit(model, calib_dataloader)

    preds = None
    labels = None

    # Test non-MV accuracy
    for x, y_true in test_dataloader:
        logits = model(x).cpu().detach().numpy()
        y = np.argmax(logits, axis=1)
        y_true = y_true.cpu().detach().numpy()
        if preds is None:
            preds = y
            labels = y_true
        else:
            preds = np.hstack((preds, y))
            labels = np.hstack((labels, y))

    # Test majority vote accuracy
    label_majority_voted = emv.majority_vote(labels, n_votes)
    pred_majority_voted = emv.majority_vote(preds, n_votes)

    raw_acc = accuracy_score(labels, preds)
    majority_acc = accuracy_score(label_majority_voted, pred_majority_voted)

    return {
        "acc_raw": [raw_acc],
        "acc_maj": [majority_acc],
    }


def test_cnn_pop_layer(
    model: etm.EmagerCNN,
    test_dataloader: DataLoader,
    calib_dataloader: None | DataLoader,
    transform: str,
):
    n_votes = 150 // etrans.get_transform_decimation(transform)

    model.set_finetune(True)
    fe = model.fe
    classi = LinearDiscriminantAnalysis()

    logits_list = []
    labels_list = []
    for batch in calib_dataloader:
        # Assuming batch is a tuple (features, labels)
        features, labels = batch
        logits = fe(features).detach()
        # Convert to NumPy and store
        logits_list.append(logits.numpy())  # Convert tensor to NumPy
        labels_list.append(labels.numpy())

    # Concatenate all batches into single NumPy arrays
    features_np = np.concatenate(logits_list, axis=0)
    labels_np = np.concatenate(labels_list, axis=0)

    classi.fit(features_np, labels_np)

    preds = None
    labels = None

    # Test non-MV accuracy
    for x, y_true in test_dataloader:
        logits = fe(x).cpu().detach().numpy()
        y = classi.predict(logits)
        y_true = y_true.cpu().detach().numpy()
        if preds is None:
            preds = y
            labels = y_true
        else:
            preds = np.hstack((preds, y))
            labels = np.hstack((labels, y))

    # Test majority vote accuracy
    label_majority_voted = emv.majority_vote(labels, n_votes)
    pred_majority_voted = emv.majority_vote(preds, n_votes)

    raw_acc = accuracy_score(labels, preds)
    majority_acc = accuracy_score(label_majority_voted, pred_majority_voted)

    return {
        "acc_raw": [raw_acc],
        "acc_maj": [majority_acc],
    }


def train_cnn(data_root, subject, train_session, val_reps, transform, quant):
    if isinstance(transform, str):
        transform = etrans.transforms_lut[g.TRANSFORM]

    # Boilerplate
    trainer = L.Trainer(
        accelerator="auto" if torch.cuda.is_available() or quant == -1 else "cpu",
        callbacks=[EarlyStopping(monitor="train_loss")],
        enable_checkpointing=False,
        logger=False,
        max_epochs=10,
    )
    train, test_intra = etd.get_lnocv_dataloaders(
        data_root, subject, train_session, val_reps, transform=transform
    )
    test_inter, calib_inter = etd.get_lnocv_dataloaders(
        data_root,
        subject,
        2 if int(train_session) == 1 else 1,
        val_reps,
        absda="none",
        transform=transform,
    )

    # Train and test
    model = etm.EmagerCNN((4, 16), 6, quant)
    trainer.fit(model, train)

    intra_test_results = test_cnn_pop_layer(model, test_intra, train, transform)
    inter_test_results = test_cnn_pop_layer(model, test_inter, calib_inter, transform)

    # intra_test_results = test_cnn(model, trainer, test_intra, None, transform)
    # inter_test_results = test_cnn(model, trainer, test_inter, calib_inter, transform)
    test_results = pd.DataFrame(
        {
            "shots": [-1],  # patch
            "acc_raw_intra": intra_test_results["acc_raw"],
            "acc_maj_intra": intra_test_results["acc_maj"],
            "acc_raw_inter": inter_test_results["acc_raw"],
            "acc_maj_inter": inter_test_results["acc_maj"],
        }
    )
    return model, test_results


def train_all_cnn(cross_validations: list[str], quantizations: list[int], transform):
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
                for quant in quantizations[q_start:]:
                    print("*" * 100)
                    print(f"Current datetime: {datetime.now()}")
                    print(
                        f"Training subject {subj} on session {ses} with L{len(valid_reps)}OCV reps={valid_reps} with {quant}-bit quantization."
                    )
                    print("*" * 100)
                    model, test_results = train_cnn(
                        g.EMAGER_DATASET_ROOT, subj, ses, valid_reps, transform, quant
                    )
                    utils.save_model(model, test_results, subj, ses, valid_reps, quant)


def test_all_cnn(cross_validations: list[str], quantizations: list[int], transform):
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

                    trainer = L.Trainer(
                        accelerator=(
                            "auto"
                            if torch.cuda.is_available() or quant == -1
                            else "cpu"
                        ),
                        enable_checkpointing=False,
                        logger=False,
                    )
                    train, test_intra = etd.get_lnocv_dataloaders(
                        g.EMAGER_DATASET_ROOT,
                        SUBJECT,
                        ses,
                        valid_reps,
                        transform=transform,
                    )
                    calib_inter, test_inter = etd.get_lnocv_dataloaders(
                        g.EMAGER_DATASET_ROOT,
                        SUBJECT,
                        "001" if ses == "002" else "002",
                        valid_reps,
                        absda="none",
                        transform=transform,
                    )
                    model = etm.EmagerCNN((4, 16), 6, quant)
                    model = utils.load_model(model, subj, ses, valid_reps, quant)
                    intra_test_results = test_cnn(
                        model, trainer, test_intra, train, transform
                    )
                    inter_test_results = test_cnn(
                        model, trainer, test_inter, calib_inter, transform
                    )
                    test_results = pd.DataFrame(
                        {
                            "acc_raw_intra": intra_test_results["acc_raw"],
                            "acc_maj_intra": intra_test_results["acc_maj"],
                            "acc_raw_inter": inter_test_results["acc_raw"],
                            "acc_maj_inter": inter_test_results["acc_maj"],
                        }
                    )
                    utils.save_model(model, test_results, subj, ses, valid_reps, quant)


if __name__ == "__main__":
    L.seed_everything(310)

    cross_validations = list(zip(ed.get_repetitions()[::2], ed.get_repetitions()[1::2]))
    # quantizations = [1, 2, 3, 4, 6, 8, 32]

    SUBJECT = 13
    SESSION = 2
    VALID_REPS = [1, 8]
    QUANT = 4

    # # for ses in [1, 2]:
    # for ses in [SESSION]:
    #     # for reps in cross_validations:
    #     for reps in [VALID_REPS]:
    #         # for q in [1, 2, 3, 4, 6, 8]:
    #         for q in [QUANT]:
    #             model, results = train_cnn(
    #                 g.EMAGER_DATASET_ROOT, SUBJECT, ses, reps, etrans.root_processing, q
    #             )
    #             print(results)
    #             utils.save_model(model, results, SUBJECT, ses, reps, q)

    model, results = train_cnn(
        g.EMAGER_DATASET_ROOT,
        SUBJECT,
        SESSION,
        VALID_REPS,
        etrans.root_processing,
        QUANT,
    )
    print(results)

    # utils.save_model(model, results, SUBJECT, SESSION, VALID_REPS, QUANT)

    # trainer = L.Trainer(
    #     accelerator="auto",
    #     enable_checkpointing=False,
    #     logger=False,
    # )
    # train, test_intra = etd.get_lnocv_dataloaders(
    #     g.EMAGER_DATASET_ROOT,
    #     SUBJECT,
    #     SESSION,
    #     VALID_REPS,
    #     transform=etrans.root_processing,
    # )
    # calib_inter, test_inter = etd.get_lnocv_dataloaders(
    #     g.EMAGER_DATASET_ROOT,
    #     SUBJECT,
    #     1 if SESSION == 2 else 2,
    #     VALID_REPS,
    #     absda="none",
    #     transform=etrans.root_processing,
    # )
    # print(
    #     test_cnn(
    #         utils.load_model(
    #             etm.EmagerCNN((4, 16), 6, QUANT), SUBJECT, SESSION, VALID_REPS, QUANT
    #         ),
    #         trainer,
    #         test_inter,
    #         None,
    #         etrans.root_processing,
    #     )
    # )
    # print(
    #     test_cnn_pop_layer(
    #         utils.load_model(
    #             etm.EmagerCNN((4, 16), 6, QUANT), SUBJECT, SESSION, VALID_REPS, QUANT
    #         ),
    #         test_inter,
    #         calib_inter,
    #         etrans.root_processing,
    #     )
    # )
