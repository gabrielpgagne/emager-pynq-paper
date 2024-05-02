"""
This module is used to train all subjects with the SCNN model and do preliminary testing.
"""

import numpy as np
import pandas as pd

import torch.cuda
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import accuracy_score

import emager_py.dataset as ed
import emager_py.data_processing as dp
import emager_py.utils as eu
import emager_py.torch.models as etm
import emager_py.transforms as etrans
import emager_py.torch.datasets as etd
import emager_py.torch.utils as etu
import emager_py.majority_vote as emv

import utils

from globals import EMAGER_DATASET_ROOT


def train_scnn(
    subject,
    train_session,
    val_rep,
    quant,
    transform,
    image_shape=(4, 16),
    max_epoch=15,
):
    # Boilerplate
    train, val, test = etd.get_triplet_dataloaders(
        EMAGER_DATASET_ROOT,
        subject,
        train_session,
        val_rep,
        transform=transform,
    )
    model = etm.EmagerSCNN(image_shape, quant)
    trainer = L.Trainer(
        accelerator="auto" if torch.cuda.is_available() or quant == -1 else "cpu",
        max_epochs=max_epoch,
        callbacks=[EarlyStopping(monitor="val_loss")],
        enable_checkpointing=False,
        logger=False,
    )

    # Train the model
    trainer.fit(model, train, val)

    # Test the model and save n-shot accuracy
    test_dict = {
        "shots": [],
        "acc_raw": [],
        "acc_maj": [],
    }

    n_votes = 150 // eu.get_transform_decimation(transform)

    embeddings, labels = etu.get_all_embeddings(model, test, model.device)
    majority_votes_true = emv.majority_vote(labels, n_votes)

    for i in [1, 2, 3, 5, 7, 10, 13, 15, 18, 20, -1]:
        n_shots_embeddings = dp.get_n_shot_embeddings(embeddings, labels, 6, i)
        model.set_target_embeddings(n_shots_embeddings)

        # Test and return accuracy
        test_ret = trainer.test(model, test)

        raw_acc = accuracy_score(model.test_preds, labels, normalize=True)

        # Test majority vote accuracy
        majority_votes_pred = emv.majority_vote(model.test_preds, n_votes)
        majority_acc = accuracy_score(
            majority_votes_true, majority_votes_pred, normalize=True
        )

        # Capture results
        for k, test_val in test_ret[0].items():
            if k not in test_dict:
                test_dict[k] = []
            test_dict[k].append(test_val)

        test_dict["shots"].append(i)
        test_dict["acc_raw"].append(raw_acc)
        test_dict["acc_maj"].append(majority_acc)

    results_df = pd.DataFrame(test_dict)

    # Save model and test outputs
    utils.save_model(model, results_df, subject, train_session, val_rep, quant)

    return model, results_df


def train_all_scnn(quantizations, transform, image_shape=(4, 16), max_epoch=15):
    models = []
    test_outputs = []

    if not isinstance(quantizations, list):
        quantizations = [quantizations]

    valid_reps_pairs = list(zip(ed.get_repetitions()[::2], ed.get_repetitions()[1::2]))
    for subj in ed.get_subjects(EMAGER_DATASET_ROOT):
        for ses in ed.get_sessions():
            for valid_reps in valid_reps_pairs:
                for quant in quantizations:
                    print("*" * 80)
                    print(
                        f"Training subject {subj} on session {ses} with L2OCV reps={valid_reps} with {quant}-bit quantization."
                    )
                    print("*" * 80)

                    model, test_out = train_scnn(
                        subj,
                        ses,
                        list(valid_reps),
                        quant,
                        transform,
                        image_shape,
                        max_epoch,
                    )
                    models.append(model)
                    test_outputs.append(test_out)
                    print(f"Test output: {test_out}")
    return models, test_outputs


if __name__ == "__main__":
    quantizations = [1, 2, 3, 4, 6, 8, 32]
    train_all_scnn(quantizations, etrans.default_processing, max_epoch=10)
    # TODO : convert to FINN-ONNX and build
