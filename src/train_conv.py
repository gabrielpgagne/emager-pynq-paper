"""
This module is used to train all subjects with the CNN model and do preliminary testing.
"""

from matplotlib import pyplot as plt
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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
        model.fe.eval()
        model.classifier.train()
        trainer.fit(model, calib_dataloader)
    model.eval()

    # Test post-calibration if needed
    test_labels = []
    test_preds = []
    for x, y_true in test_dataloader:
        with torch.no_grad():
            logits = model(x).cpu().detach().numpy()
        y = np.argmax(logits, axis=1)
        y_true = y_true.cpu().detach().numpy()

        test_labels.extend(y_true)
        test_preds.extend(y)

    test_labels = np.array(test_labels)
    test_preds = np.array(test_preds)

    # Test majority vote accuracy
    label_majority_voted = emv.majority_vote(test_labels, n_votes)
    pred_majority_voted = emv.majority_vote(test_preds, n_votes)

    raw_acc = accuracy_score(test_labels, test_preds)
    majority_acc = accuracy_score(label_majority_voted, pred_majority_voted)

    cm = confusion_matrix(test_labels, test_preds, normalize="true")
    cm2 = confusion_matrix(label_majority_voted, pred_majority_voted, normalize="true")

    return {
        "acc_raw": [raw_acc],
        "acc_maj": [majority_acc],
        "conf_mat_raw": [cm],
        "conf_mat_maj": [cm2],
    }


def test_cnn_pop_layer(
    model: etm.EmagerCNN,
    test_dataloader: DataLoader,
    calib_dataloader: None | DataLoader,
    transform: str,
):
    n_votes = 150 // etrans.get_transform_decimation(transform)

    fe = nn.Sequential(*([mod for mod in model.modules()][:-1]))
    fe.eval()
    classi = LinearDiscriminantAnalysis()

    calib_logits = []
    calib_labels = []
    for batch in calib_dataloader:
        # Assuming batch is a tuple (features, labels)
        features, labels = batch
        with torch.no_grad():
            logits = fe(features).detach()
        calib_logits.extend(logits.numpy())
        calib_labels.extend(labels.numpy())

    # Concatenate all batches into single NumPy arrays
    calib_logits = np.array(calib_logits)
    calib_labels = np.array(calib_labels)

    classi.fit(calib_logits, calib_labels)

    # Test post-calibration if needed
    test_labels = []
    test_preds = []
    for x, y_true in test_dataloader:
        with torch.no_grad():
            logits = fe(x).cpu().detach().numpy()
        y = classi.predict(logits)
        y_true = y_true.cpu().detach().numpy()

        test_labels.extend(y_true)
        test_preds.extend(y)

    test_labels = np.array(test_labels)
    test_preds = np.array(test_preds)

    # Test majority vote accuracy
    label_majority_voted = emv.majority_vote(test_labels, n_votes)
    pred_majority_voted = emv.majority_vote(test_preds, n_votes)

    raw_acc = accuracy_score(test_labels, test_preds)
    majority_acc = accuracy_score(label_majority_voted, pred_majority_voted)

    cm = confusion_matrix(test_labels, test_preds, normalize="true")
    cm2 = confusion_matrix(label_majority_voted, pred_majority_voted, normalize="true")

    return {
        "acc_raw": [raw_acc],
        "acc_maj": [majority_acc],
        "conf_mat_raw": [cm],
        "conf_mat_maj": [cm2],
    }


def train_cnn(data_root, subject, train_session, val_reps, transform, quant):
    if isinstance(transform, str):
        transform = etrans.transforms_lut[g.TRANSFORM]

    # Boilerplate
    trainer = L.Trainer(
        accelerator="auto" if torch.cuda.is_available() or quant == -1 else "cpu",
        callbacks=[EarlyStopping(monitor="val_loss")],
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
        shuffle="test",
        transform=transform,
        train_batch=256,
        test_batch=64,
    )

    # Train and test
    model = etm.EmagerCNN((4, 16), 6, quant)
    trainer.fit(model, train, test_intra)

    # intra_test_results = test_cnn_pop_layer(model, test_intra, train, transform)
    # inter_test_results = test_cnn_pop_layer(model, test_inter, calib_inter, transform)

    intra_test_results = test_cnn(model, trainer, test_intra, None, transform)
    inter_test_results = test_cnn(model, trainer, test_inter, calib_inter, transform)

    test_results = pd.DataFrame(
        {
            "shots": [-1],
            #
            "acc_raw_intra": intra_test_results["acc_raw"],
            "acc_maj_intra": intra_test_results["acc_maj"],
            "conf_mat_raw_intra": intra_test_results["conf_mat_raw"],
            "conf_mat_maj_intra": intra_test_results["conf_mat_maj"],
            #
            "acc_raw_inter": inter_test_results["acc_raw"],
            "acc_maj_inter": inter_test_results["acc_maj"],
            "conf_mat_raw_inter": inter_test_results["conf_mat_raw"],
            "conf_mat_maj_inter": inter_test_results["conf_mat_maj"],
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
    torch.set_float32_matmul_precision("high")

    cross_validations = list(zip(ed.get_repetitions()[::2], ed.get_repetitions()[1::2]))
    # quantizations = [1, 2, 3, 4, 6, 8, 32]

    SUBJECT = 14
    SESSION = 1
    VALID_REPS = [1]
    QUANT = 8

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
    print(
        results["acc_raw_intra"].values[0],
        results["acc_maj_intra"].values[0],
        results["acc_raw_inter"].values[0],
        results["acc_maj_inter"].values[0],
    )
    utils.save_model(model, results, SUBJECT, SESSION, VALID_REPS, QUANT)

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
