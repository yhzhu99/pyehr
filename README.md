# PyEHR: A Predictive Modeling Toolkit for Electronic Health Records

This repository is the official implementation of our paper, ["A Comprehensive Benchmark for COVID-19 Predictive Modeling Using Electronic Health Records in Intensive Care"](https://www.cell.com/patterns/fulltext/S2666-3899(24)00050-3), which was accepted by *Patterns*.
*\*Equal contribution*

The repository includes various machine learning and deep learning models implemented for predictive modeling tasks using Electronic Health Records (EHR) specifically for COVID-19 patients in Intensive Care Units (ICU).

Benchmarking results from two real-world COVID-19 EHR datasets (TJH and CDSL datasets) are also provided. All results and trained models are freely accessible on our online platform [PyEHR](https://pyehr.netlify.app), and the source code can be found [here](https://github.com/yhzhu99/pyehr-playground).

We invite clinicians, researchers, and data scientists to contribute to this growing platform.

## 🎯 Prediction Tasks

The following prediction tasks have been implemented in this repository:

- [x] Mortality outcome prediction (Early)
- [x] Length-of-stay prediction
- [x] Multi-task/Two-stage prediction (predict mortality outcome and length-of-stay simultaneously)

## 🚀 Model Zoo

The repository contains a variety of models from traditional machine learning, basic deep learning, and advanced deep learning models tailored for EHR data:

### Machine Learning Models

- [x] Random forest (RF)
- [x] Decision tree (DT)
- [x] Gradient Boosting Decision Tree (GBDT)
- [x] XGBoost
- [x] CatBoost

### Deep Learning Models

- [x] Multi-layer perceptron (MLP)
- [x] Recurrent neural network (RNN)
- [x] Long-short term memory network (LSTM)
- [x] Gated recurrent units (GRU)
- [x] Temporal convolutional networks
- [x] Transformer

### EHR Predictive Models

- [x] RETAIN
- [x] StageNet
- [x] Dr. Agent
- [x] AdaCare
- [x] ConCare
- [x] GRASP

The best searched hyperparameters for each model are meticulously preserved in the configs/ folder (`dl.py` and `ml.py`).

## 🗄️ Repository Structure

The code repository includes the following directory structure:

```bash
pyehr/
├── losses/ # contains losses designed for the tasks
├── metrics/ # contains metrics for tasks
├── models/ # backbone models ML or DL models
├── configs/ # contains configs of best searched hyperparameters and dataset related configs
├── datasets/ # contains datasets and pre-process scripts
├── pipelines/ # deep learning or machine learning pipeline under pytorch lightning framework
├── tune.py # do hyper-parameter search with WandB
├── train.py # train models
├── test.py # test the models
└── pyproject.toml # project dependencies for uv
```

## 🗂️ Data Format

The inputs fed to the pipelines should have the following data format:

- `x.pkl`: (N, T, D) List, where N is the number of patients, T is the number of time steps, and D is the number of features. At D dimension, the first x features are demographic features, the next y features are lab test features, where x + y = D
- `y.pkl`: (N, T, 2) List, where the 2 values are [outcome, length-of-stay] for each time step.
- `los_info.pkl`: a dictionary contains length-of-stay related statatistics. E.g. mean and std of the los values. Since we have done z-score normalization to the los labels, these stats are essential to reverse the raw los values.

## ⚙️ Requirements

This project uses [uv](https://docs.astral.sh/uv/) to manage a reproducible Python 3.12 environment.

```bash
uv sync
```

## 📈 Usage

To start with the data pre-precessing steps, follow the instructions:

1. Download TJH dataset from paper [An interpretable mortality prediction model for COVID-19 patients](https://www.nature.com/articles/s42256-020-0180-7), and you need to apply for the CDSL dataset if necessary. [Covid Data Save Lives Dataset](https://www.hmhospitales.com/prensa/notas-de-prensa/comunicado-covid-data-save-lives)
3. Run the pre-processing scripts `preprocess_{dataset}.ipynb` in `datasets/` folder.
4. Then you will have the 10-fold processed datasets in the required data format.

To start with the training or testing, use the following commands:

```bash
# Hyperparameter tuning
uv run python dl_tune.py # for deep learning models
uv run python ml_tune.py # for machine learning models

# Model training
uv run python train.py

# Model testing
uv run python test.py
```

## 📜 License

This project is licensed under the terms of the MIT license. See [LICENSE](LICENSE) for additional details.

## 🙏 Contributors

This project is brought to you by the following contributors:

- [Yinghao Zhu](https://github.com/yhzhu99)
- [Wenqing Wang](https://github.com/ericaaaaaaaa)
- [Junyi Gao](https://github.com/v1xerunt)
- [Liantao Ma](https://github.com/massltime)

For a deeper dive into our research, please refer to our *Patterns* [paper](https://www.cell.com/patterns/fulltext/S2666-3899(24)00050-3).

```
@article{gao2024comprehensive,
  title={A comprehensive benchmark for COVID-19 predictive modeling using electronic health records in intensive care},
  author={Gao, Junyi and Zhu, Yinghao and Wang, Wenqing and Wang, Zixiang and Dong, Guiying and Tang, Wen and Wang, Hao and Wang, Yasha and Harrison, Ewen M and Ma, Liantao},
  journal={Patterns},
  volume={5},
  number={4},
  year={2024},
  publisher={Elsevier}
}
```
