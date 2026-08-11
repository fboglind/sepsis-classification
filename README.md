# Sepsis Classification from ICU Time-Series Data

This project classifies patients as septic or non-septic using multivariate ICU time-series data. It compares attention-based bidirectional LSTM models across 2-, 4-, and 6-hour classification horizons.

The sepsis label is constant for each patient, so the task is patient-level classification rather than estimation of the precise time of sepsis onset.

## Data

The data is divided into four SepsisExp partitions. Partitions A–C are combined for training, while partition D is reserved for testing.

- Training set: 957 patients, including 222 septic patients (23.2%)
- Test set: 318 patients, including 74 septic patients (23.3%)
- Input: 43 clinical features, including vital signs, laboratory measurements, and demographic information

## Method

The classifier consists of a two-layer bidirectional LSTM with an attention mechanism and a feed-forward classification head. Training uses Adam, binary cross-entropy loss, class weighting, learning-rate scheduling, and early stopping.

Feature importance is examined using leave-one-feature-out ablation and attention-weight visualization.

## Results

| Horizon | ROC AUC | Accuracy | Recall | F1 score |
| --- | ---: | ---: | ---: | ---: |
| 2 hours | 0.7370 | 0.7591 | 0.4216 | 0.4489 |
| 4 hours | 0.7261 | 0.7654 | 0.4108 | 0.4490 |
| 6 hours | 0.7099 | 0.7748 | 0.4757 | 0.4958 |

The 2-hour model achieved the highest ROC AUC, while the 6-hour model achieved the highest recall and F1 score. The latter therefore provided the best balance for identifying septic patients at the default classification threshold.

Ablation analysis indicated that the most important features varied by horizon. Important variables included leukocytes, heart rate, lactate, respiratory minute volume, cardiovascular measurements, and blood-gas measures.

## Requirements

The project uses Python 3.10 and the following main packages:

```bash
python -m pip install jupyter pandas numpy torch scikit-learn matplotlib seaborn
```

The notebook also requires these project modules:

```text
model.py
custom_datasets.py
training_pipeline.py
utility_functions.py
```

Place the four TSV data partitions in `raw_data/`. The notebook expects pretrained models in `models/` and saved results in `processed_data/`. These paths can be changed through `DATA_DIR`, `MODELS_DIR`, and `OUT_DIR` near the beginning of the notebook.

Open the notebook in Jupyter and run its cells in order. By default, it loads pretrained models. The training and ablation cells can be uncommented to reproduce those experiments.

## Limitations

- The models were evaluated on a single held-out partition and have not been externally validated.
- Class imbalance makes accuracy insufficient on its own; recall, F1 score, and ROC AUC should be considered together.
- Attention weights and ablation results indicate model sensitivity, but do not establish clinical causality.
- This is an educational project and must not be used for clinical decision-making.

