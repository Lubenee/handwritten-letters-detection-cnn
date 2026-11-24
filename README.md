# Letter Recognition CNN

A Keras CNN that recognizes handwritten letters. The end goal is to recognize both **Cyrillic** and **Latin** letters. You draw a letter in an OpenCV window and the model predicts which one it is.

## Usage

Requires Python 3.14.

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py
```

If no trained model (`.keras`) is found, a new one is trained and saved. Training results are written to `Logs/`.

## Datasets

- **Latin**: [EMNIST Balanced](https://www.nist.gov/itl/products-and-services/emnist-dataset), loaded through [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/emnist) (`emnist/balanced`, digits removed)
- **Cyrillic**: [CMNIST](https://github.com/bolattleubayev/cmnist) 
If something is wrong with the archive, make sure to clone this repo inside the 'cmnist' folder in root.
(Cyrillic MNIST), cached locally as `cmnist_cyrilic.npz`
