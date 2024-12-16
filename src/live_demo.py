import lightning as L
import torch.cuda
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
import threading

from emager_py.finn import remote_operations as ro
import emager_py.screen_guided_training as sgt
import emager_py.emager_redis as er
import emager_py.data_processing as dp
import emager_py.dataset as ed
import emager_py.transforms as etrans
import emager_py.torch.models as etm
from emager_py import utils

import globals as g
from finn_build import launch_finn_build

import time
import queue
from tkinter import Tk, Label, Button
import tkinter.ttk as ttk
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
                "python3 client.py",
            ),
            args=(),
            daemon=True,
        )
        t.start()

        preds_buffer = []
        while t.is_alive():
            pred_bytes = r.pop_fifo(r.PREDICTIONS_FIFO_KEY, timeout=1)
            pred = r.decode_label_bytes(pred_bytes)
            print(f"Received predictions: {pred}")

            preds_buffer.append(pred.item(0))
            if len(preds_buffer) >= 10:  # 25 ms per prediction -> 250 ms
                pred_mv = np.argmax(np.bincount(preds_buffer))  # majority vote
                preds_buffer = []
                id_queue.put(pred_mv)

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


if __name__ == "__main__":
    utils.set_logging()

    # hostname = er.get_docker_redis_ip()
    hostname = "pynq.local"
    images_path = "output/gestures/"

    subject = 13
    session = 1
    quant = 8
    shots = 10
    transform = etrans.transforms_lut[g.TRANSFORM]
    finetune_data_dir = "data/finetune/"
    n_reps = 1
    rep_time = 3

    r = er.EmagerRedis(hostname)
    r.set_pynq_params(g.TRANSFORM)
    r.set_rhd_sampler_params(
        low_bw=5,
        hi_bw=450,
        en_dsp=1,
        fp_dsp=20,
        bitstream=ro.DEFAULT_EMAGER_PYNQ_PATH + "bitfile/finn-accel.bit",
        # bitstream=""
    )
    r.clear_data()

    c = ro.connect_to_pynq(hostname=hostname)

    def resume_training_cb(gesture_id):
        ro.sample_training_data(
            c,
            hostname,
            rep_time * g.EMAGER_SAMPLING_RATE,
            g.TARGET_EMAGER_PYNQ_PATH,
            gesture_id,
        )

    # ========== TRAINING ==========

    # r.clear_data()

    # sgt.EmagerGuidedTraining(
    #     n_reps,
    #     gestures_path="output/gestures/",
    #     resume_training_callback=resume_training_cb,
    #     callback_arg="gesture",
    # ).start()

    # # raw data -> processed DataLoader
    # data = r.dump_labelled_to_numpy()
    # print(data.shape)

    # data_dir = ed.process_save_dataset(
    #     data, finetune_data_dir, transform, subject, session
    # )
    # data = ed.load_emager_data(data_dir, subject, session)

    # data, labels = dp.extract_labels_and_roll(data, 2)
    # data = data.astype(np.float32)
    # train_triplets = dp.generate_triplets(data, labels, 5000)
    # train_triplets = [
    #     torch.from_numpy(t).reshape((-1, 1, *g.EMAGER_DATA_SHAPE))
    #     for t in train_triplets
    # ]
    # train_dl = DataLoader(TensorDataset(*train_triplets), batch_size=32, shuffle=True)

    # # Train
    # trainer = L.Trainer(
    #     accelerator="auto" if torch.cuda.is_available() or quant == -1 else "cpu",
    #     enable_checkpointing=False,
    #     logger=True,
    #     max_epochs=10,
    # )
    # model = etm.EmagerSCNN(quant)
    # trainer.fit(model, train_dl)
    # launch_finn_build(subject, quant, shots, g.TRANSFORM)

    # =========== TESTING ==============

    # Get some calibration data
    USE_LIVE_DATA = True

    if USE_LIVE_DATA:
        # TODO doesnt seem to work
        sgt.EmagerGuidedTraining(
            1,
            gestures=[2, 14, 26, 1, 8, 30],
            training_time=1,
            gestures_path=images_path,
            resume_training_callback=resume_training_cb,
            callback_arg="gesture",
        ).start()
    else:
        # works!
        calib_data, test_data = ed.get_lnocv_datasets(
            g.EMAGER_DATASET_ROOT, 0, session, [0, 1]
        )
        calib_data, calib_labels = dp.extract_labels(calib_data)
        for i in np.unique(calib_labels):
            matches = np.argwhere(calib_labels == i).flatten()
            idxs = np.random.choice(
                matches, shots * g.EMAGER_SAMPLE_BATCH, replace=False
            )
            for idx in idxs:
                r.push_sample(calib_data[idx], calib_labels[idx])

    gui_readloop(hostname, images_path)
