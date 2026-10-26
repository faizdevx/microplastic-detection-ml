# Raspberry Pi 5

The code runs on CPU only. Nothing here has been run on a Raspberry Pi yet; the table at the
end stays "not run" until `python -m mp.cli benchmark --pi` is executed on one.

## Setup (64-bit Raspberry Pi OS, Python 3.10 or newer)

```bash
sudo apt update && sudo apt install -y python3-venv python3-picamera2 libopenblas0
git clone https://github.com/faizdevx/microplastic-detection-ml.git
cd microplastic-detection-ml
python3 -m venv --system-site-packages .venv   # keeps the system picamera2
source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu torch
pip install -e ".[dev,pi]"
make data audit features train-cls     # or copy data/ from another machine
```

`torch` is only needed for the YOLO pipeline; the polymer classifier and `mp analyse` need
scikit-learn, OpenCV and NumPy.

## Capture a frame with fixed settings

```python
from mp.deploy.capture import capture
capture("sample_001.jpg", exposure_us=20000, analogue_gain=1.0, notes="Nile Red, 365 nm")
```

This writes `sample_001.jpg` and `sample_001.jpg.json` with exposure, gain and the camera
metadata. Use the same settings for every sample and keep the sidecar files: the dataset
audit shows that image brightness and size differ between the two classes.

## Analyse

```bash
python -m mp.cli analyse --image sample_001.jpg
python -m mp.cli analyse --image sample_001.jpg --sensors 160 70 64 64 --json
```

Camera frames must be 4:3 and have a resolution that appears in the training data
(for example 640x480), otherwise the polymer label is withheld (`docs/api.md`).

## Benchmark

```bash
python -m mp.cli benchmark --pi      # refuses to run on anything that is not a Raspberry Pi
```

Writes `results/deploy_benchmark_pi.json` (median and p95 latency over 50 runs after warm-up,
peak memory, throughput, device and library versions).

## Results on a Raspberry Pi

| Device | Median ms | p95 ms | Peak RSS MB | Images/s |
|---|---|---|---|---|
| Raspberry Pi 5 | not run | not run | not run | not run |
