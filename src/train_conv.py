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
    decim: int,
):
    n_votes = 150 // decim

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
    decim: str,
):
    n_votes = 150 // decim

    model.classifier = nn.Identity()
    model.fe.eval()
    classi = LinearDiscriminantAnalysis()

    calib_logits = []
    calib_labels = []
    for batch in calib_dataloader:
        # Assuming batch is a tuple (features, labels)
        features, labels = batch
        with torch.no_grad():
            logits = model(features).detach()
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
            logits = model(x).cpu().detach().numpy()
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


def train_cnn(
    data_root,
    subject,
    train_session,
    val_reps,
    test_reps,
    transform,
    quant,
    save=False,
    ws=1,
):
    if not quant:
        return

    if isinstance(transform, str):
        transform = etrans.transforms_lut[transform]

    # Boilerplate
    trainer = L.Trainer(
        accelerator="auto" if torch.cuda.is_available() or quant == -1 else "cpu",
        callbacks=[EarlyStopping(monitor="val_loss", min_delta=0.01, mode="min")],
        enable_checkpointing=False,
        logger=False,
        max_epochs=10,
    )

    train, val_intra = etd.get_lnocv_dataloaders(
        data_root, subject, train_session, val_reps, transform=transform, ws=ws
    )

    test_intra = etd.get_lnocv_dataloaders(
        data_root, subject, train_session, test_reps, transform=transform, ws=ws
    )[1]

    inter_session = 2 if int(train_session) == 1 else 1

    _, calib_inter = etd.get_lnocv_dataloaders(
        data_root,
        subject,
        inter_session,
        val_reps,
        absda="none",
        shuffle="test",
        transform=transform,
        test_batch=64,
        ws=ws,
    )

    _, test_inter = etd.get_lnocv_dataloaders(
        data_root,
        subject,
        inter_session,
        test_reps,
        absda="none",
        shuffle="none",
        transform=transform,
        test_batch=256,
        ws=ws,
    )

    # Train and test
    model = etm.EmagerCNN((4, 16), 6, quant, ws)
    trainer.fit(model, train, val_intra)

    # intra_test_results = test_cnn_pop_layer(model, test_intra, train, transform)
    # inter_test_results = test_cnn_pop_layer(model, test_inter, calib_inter, transform)

    decim = etrans.get_transform_decimation(transform) * ws

    intra_test_results = test_cnn(model, trainer, test_intra, None, decim)
    inter_test_results = test_cnn(model, trainer, test_inter, calib_inter, decim)

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

    if save:
        utils.save_model(
            model, test_results, subject, train_session, val_reps + test_reps, quant
        )

    return model, test_results


def train_all_cnn(
    cross_validations: list[str], quantizations: list[int], transform, ws
):
    import multiprocessing as mp
    from multiprocessing import Process
    import time

    num_procs = 4
    mp.set_start_method("spawn")

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

                    for q in quants:
                        p = Process(
                            target=train_cnn,
                            args=(
                                g.EMAGER_DATASET_ROOT,
                                subj,
                                int(ses),
                                valid_reps[0],
                                valid_reps[1],
                                transform,
                                q,
                                True,
                                ws,
                            ),
                        )
                        p.start()
                        procs.append(p)
                        time.sleep(10)

                    for p in procs:
                        p.join()


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

    # ============ Train all models ==========

    # cross_validations = list(zip(ed.get_repetitions()[::2], ed.get_repetitions()[1::2]))
    # cross_validations = [("002", "008")]
    # quantizations = [1, 2, 3, 4, 6, 8, 32]

    # train_all_cnn(cross_validations, quantizations, etrans.root_processing)

    # ============== Test all models =============

    # ========= Train a single model ==========

    SUBJECT = 8
    SESSION = 1

    TEST_REPS = [2]
    VAL_REPS = [9]

    QUANT = 8

    model, results = train_cnn(
        g.EMAGER_DATASET_ROOT,
        SUBJECT,
        SESSION,
        VAL_REPS,
        TEST_REPS,
        # etrans.root_processing,
        etrans.filter_rect_u8_processing,
        # etrans.filter_rect_processing,
        # etrans.default_processing,
        QUANT,
        ws=25,
        # ws=1,
    )

    print(
        results["acc_raw_intra"].values[0],
        results["acc_maj_intra"].values[0],
        results["acc_raw_inter"].values[0],
        results["acc_maj_inter"].values[0],
    )
    utils.save_model(model, results, SUBJECT, SESSION, VAL_REPS + TEST_REPS, QUANT)

    # ========= Test a single model ==========

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
