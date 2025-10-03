import lightning as L
import torch.cuda
import threading
import pandas as pd
import json
import os

from emager_py.finn import remote_operations as ro
import emager_py.emager_redis as er
import emager_py.data_processing as dp
import emager_py.dataset as ed
import emager_py.transforms as etrans
import emager_py.torch.models as etm
import emager_py.torch.datasets as etd

import globals as g
from finn_build_conv import launch_finn_build
import utils
import sample_data as sd

import queue
from tkinter import Tk, Label, Button
from PIL import Image, ImageTk


def gui_readloop(hostname, images_path):
    id_queue = queue.Queue()

    def generate_ids():
        r = er.EmagerRedis(hostname)
        r.set_sampling_params(n_samples=1e6)
        t = threading.Thread(
            target=lambda: ro.run_remote_finn(
                ro.connect_to_pynq(hostname=hostname),
                g.TARGET_EMAGER_PYNQ_PATH,
                "python3 client_conv.py",
            ),
            args=(),
            daemon=True,
        )
        t.start()

        while t.is_alive():
            pred_bytes = r.pop_fifo(r.PREDICTIONS_FIFO_KEY, timeout=1)
            pred = r.decode_label_bytes(pred_bytes).item(0)
            print(f"Received predictions: {pred}")
            id_queue.put(pred)

    images_ids = [
        "Hand_Close",
        "Thumbs_Up",
        "Chuck_Grip",
        "No_Motion",
        "Index_Pinch",
        "Index_Extension",
    ]
    image_files = {
        i: Image.open(f"{images_path}/{p}.png").resize((640, 320))
        for i, p in enumerate(images_ids)
    }

    # Function to update the image in the GUI
    def update_image():
        while not id_queue.empty():
            new_id = id_queue.get()
            image = image_files[new_id]
            tk_image = ImageTk.PhotoImage(image)
            label.config(image=tk_image)
            label.image = tk_image  # Prevent garbage collection
            id_label.config(text=f"{images_ids[new_id]} (ID {new_id})")
        root.after(100, update_image)  # Schedule the function to run again

    # Create the Tkinter GUI
    root = Tk()
    root.title("EMaGerZ Prediction Viewer")

    # Label to display the image
    label = Label(root)
    label.pack(pady=20)

    # Label to display the current ID
    id_label = Label(root, text="Waiting for remote device...", font=("Arial", 14))
    id_label.pack(pady=10)

    # Button to quit the application
    quit_button = Button(root, text="Quit", command=root.destroy)
    quit_button.pack(pady=10)

    # Start the ID generation thread
    thread = threading.Thread(target=generate_ids, daemon=True)
    thread.start()

    # Start updating the GUI# Start updating the GUI\update_image()
    update_image()

    # Run the Tkinter event loop
    root.mainloop()


def live_training(
    r: er.EmagerRedis,
    hostname: str,
    data_dir: str,
    subject: int,
    session: int,
    quant: int,
    shots: int,
    sample_data: bool,
    transform: str,
    n_reps=5,
    rep_time=5,
    gestures=g.GESTURES,
    img_dir: str = g.GESTURES_PATH,
):
    cross_val_rep = [1]
    if sample_data:
        sd.sample_sgt(
            r,
            hostname,
            data_dir,
            subject,
            session,
            n_reps,
            rep_time,
            gestures,
            img_dir,
        )

    train_dl, val_dl = etd.get_lnocv_dataloaders(
        data_dir, subject, session, cross_val_rep, transform=transform
    )

    # Train
    trainer = L.Trainer(
        accelerator="auto" if torch.cuda.is_available() or quant == -1 else "cpu",
        enable_checkpointing=False,
        logger=True,
        max_epochs=10,
    )
    model = etm.EmagerCNN(g.EMAGER_DATA_SHAPE, 6, quant)
    trainer.fit(model, train_dl, val_dl)

    utils.save_model(model, pd.DataFrame(), subject, session, cross_val_rep, quant)
    os.makedirs(g.OUT_DIR_ROOT + g.OUT_DIR_FINN, exist_ok=True)

    params_dict = {
        "subject": subject,
        "quantization": quant,
        "shots": shots,
        "session": session,
        "repetition": cross_val_rep,
        "transform": g.TRANSFORM,
    }

    with open(g.OUT_DIR_ROOT + g.OUT_DIR_FINN + "finn_config.json", "w") as f:
        json.dump(params_dict, f)
    utils.lock_finn()
    launch_finn_build()

    return model


def live_testing(
    r: er.EmagerRedis,
    hostname: str,
    data_dir: str,
    subject: int,
    session: int,
    sample_data: bool,
    n_reps=1,
    rep_time=5,
    gestures=g.GESTURES,
    img_path: str = g.GESTURES_PATH,
):
    if sample_data:
        sd.sample_sgt(
            r,
            hostname,
            data_dir,
            subject,
            session,
            n_reps,
            rep_time,
            gestures,
            img_path,
        )

    calib_data = ed.load_emager_data(
        data_dir,
        subject,
        session,
    )
    calib_data[..., 14] = 0  # bad channel
    calib_data, calib_labels = dp.extract_labels(calib_data)

    # Push data and labels to redis in batches
    for i in range(0, len(calib_data), g.EMAGER_SAMPLE_BATCH):
        r.push_sample(
            calib_data[i : (i + g.EMAGER_SAMPLE_BATCH)],
            calib_labels[i],
        )
    gui_readloop(hostname, img_path)


if __name__ == "__main__":
    SUBJECT = 14
    SESSION = 1

    TRAIN = True
    SAMPLE_TRAIN_DATA = False

    TEST = False
    SAMPLE_TEST_DATA = False

    # ==== Training parameters ====
    N_TRAIN_REPS = 5
    LEN_TRAIN_REPS = 5
    QUANT = 8
    SHOTS = -1
    TRAIN_DATA_DIR = "data/live_train/"

    # ==== Testing parameters ====
    N_TEST_REPS = 1
    LEN_TEST_REPS = 3
    TEST_DATA_DIR = "data/live_test/"

    # ==== Varia parameters ====
    HOSTNAME = g.PYNQ_HOSTNAME
    GESTURES_ID = g.GESTURES
    IMAGES_DIR = g.GESTURES_PATH

    TRANSFORM = etrans.transforms_lut[g.TRANSFORM]

    # ==== Script ====
    try:
        r = er.EmagerRedis(HOSTNAME)
        r.set_pynq_params(g.TRANSFORM)
        r.set_sampling_params(1000, 25, 5000)
        r.set_rhd_sampler_params(
            low_bw=15,
            hi_bw=350,
            # en_dsp=1,
            fp_dsp=20,
            bitstream=ro.DEFAULT_EMAGER_PYNQ_PATH + "bitfile/finn-accel.bit",
        )
        r.clear_data()
    except:
        r = None

    # ========== TRAINING ==========
    if TRAIN:
        live_training(
            r,
            HOSTNAME,
            TRAIN_DATA_DIR,
            SUBJECT,
            SESSION,
            QUANT,
            SHOTS,
            SAMPLE_TRAIN_DATA,
            TRANSFORM,
            N_TRAIN_REPS,
            LEN_TRAIN_REPS,
            GESTURES_ID,
            IMAGES_DIR,
        )
    # =========== TESTING ==============
    r.clear_data()
    if TEST:
        live_testing(
            r,
            HOSTNAME,
            TEST_DATA_DIR,
            SUBJECT,
            SESSION,
            SAMPLE_TEST_DATA,
            N_TEST_REPS,
            LEN_TEST_REPS,
            GESTURES_ID,
            IMAGES_DIR,
        )
