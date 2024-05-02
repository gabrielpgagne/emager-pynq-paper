# EMaGer-PYNQ

First, setup the globals in `globals.py`.

Then, run `model_training.py`. For every subject, session, LOOCV and quantization, a model will be trained and tested with a given `emager_py.transform`. This process takes _many_ hours to run.

After, you can experiment with `stats.py` to visualize the models' performance.
