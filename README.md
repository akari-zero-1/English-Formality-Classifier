# AI Formality Checker

AI Formality Checker is a Streamlit application that classifies English sentences as **Formal** or **Informal**. It provides a simple web interface and supports three trained text-classification models:

- **RoBERTa** - the highest-accuracy option, with comparatively slower inference.
- **DistilBERT** - a smaller Transformer model that balances speed and accuracy.
- **FastText** - a lightweight, CPU-friendly option with very fast inference.

The repository also contains the data preparation, training, and evaluation notebooks used to build and compare the models on the GYAFC formality dataset.

## Features

- Classify a single English sentence as Formal or Informal.
- Display the predicted label, confidence score, and the top-two class scores.
- Select the inference model from the model selector.
- Automatically hide FastText when the current Python version cannot install the optional FastText dependency.
- Cache the loaded model with Streamlit so it is not reloaded on every interaction.
- Train and evaluate RoBERTa, DistilBERT, and FastText through the included notebooks.

## How it works

1. The application loads the selected local model through `src/predictor.py`.
2. RoBERTa and DistilBERT use Hugging Face text-classification pipelines.
3. FastText uses its native model API and maps `__label__1` to `Formal`.
4. The prediction is normalized to the labels `Formal` and `Informal`.
5. The interface displays the predicted class, confidence, and both class scores.

The classifier uses the following label mapping:

| Numeric/model label | Application label |
| --- | --- |
| `0` or `__label__0` | `Informal` |
| `1` or `__label__1` | `Formal` |

## Requirements

- Python 3.8 or newer.
- A local copy of the trained model artifacts.
- A CPU is sufficient for inference. A CUDA-enabled PyTorch installation can be used when available, especially during training.

FastText is optional. The `fasttext-wheel` dependency is installed only for Python versions below 3.13 because the current FastText package used by this project does not support Python 3.13 and newer. RoBERTa and DistilBERT remain available without FastText.

## Model files

The application expects the following directories and file:

```text
models/
├── best_roberta_formality/
├── best_distilbert_formality/
└── best_fasttext_formality.bin
```

The two Transformer directories should contain the model weights, configuration, and tokenizer files produced by Hugging Face Transformers. The FastText file must be a trained `.bin` model.

The `models/` directory is ignored by Git, so trained artifacts are not included in the repository. Place the artifacts in the paths above before starting the application. If only RoBERTa and DistilBERT are available, the application can still run with those two models.

## Installation

Open a terminal in the repository root and create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\Activate.ps1
```

Activate it on macOS or Linux:

```bash
source venv/bin/activate
```

Install the project dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the application

Run the command from the repository root so that the relative `models/` paths resolve correctly:

```bash
streamlit run app.py
```

Streamlit normally opens the application at [http://localhost:8501](http://localhost:8501). Enter an English sentence, choose a model, and select **Analyze with AI** to see the result.

For example:

```text
Input: u should totally check out that new movie lol
Output: Informal
```

## Training and evaluation workflow

The notebooks document the complete machine-learning workflow:

| Notebook | Purpose |
| --- | --- |
| `notebook/01_EDA_preprocessing.ipynb` | Explore the GYAFC data and prepare Transformer-ready data. |
| `notebook/02_RoBERTa.ipynb` | Fine-tune and save the RoBERTa classifier. |
| `notebook/03_DistilBERT.ipynb` | Fine-tune and save the DistilBERT classifier. |
| `notebook/04_FastText.ipynb` | Prepare FastText-format data and train the FastText classifier. |
| `notebook/05_Evaluation.ipynb` | Compare the trained models on the official GYAFC test set. |
| `notebook/KHDL.ipynb` | An additional end-to-end exploration and preprocessing notebook. |

The prepared datasets are stored under `data/`:

- `gyafc_dataset_train_full.csv` - combined labeled training data.
- `gyafc_ready_to_train.csv` - cleaned training data for the general preprocessing workflow.
- `gyafc_transformer_ready.csv` - cleaned data used by the Transformer notebooks.
- `gyafc_test_ready.csv` - prepared official test data.
- `fasttext_train.txt` and `fasttext_test.txt` - FastText-formatted training and test files.
- `*.src` and `*.tgt` - source and target text files used during dataset preparation.

The Transformer notebooks train on the prepared training data and evaluate against `gyafc_test_ready.csv`. The evaluation notebook reports accuracy, precision, recall, F1 score, and confusion matrices for the supported models.

### Training dependencies

The Streamlit application only needs the packages in `requirements.txt`. To run the training notebooks, additional packages may be required, including:

```bash
pip install datasets evaluate accelerate scikit-learn matplotlib seaborn jupyter
```

Training large Transformer models is substantially more resource-intensive than running inference. A CUDA-enabled environment is recommended when available.

## Project structure

```text
.
├── app.py                         # Streamlit entry point
├── requirements.txt               # Runtime dependencies
├── src/
│   └── predictor.py               # Model loading and prediction logic
├── models/                        # Local trained model artifacts (ignored by Git)
├── data/                          # Prepared datasets and FastText files
└── notebook/                      # EDA, training, and evaluation notebooks
```

## Troubleshooting

### `FileNotFoundError` when starting the app

Make sure the expected model files exist under `models/` and start Streamlit from the repository root:

```bash
streamlit run app.py
```

### FastText is not listed

FastText is disabled when it cannot be imported. Use Python 3.12 or earlier and reinstall the dependencies:

```bash
pip install -r requirements.txt
```

RoBERTa and DistilBERT do not depend on FastText.

### DistilBERT input errors

The project includes a custom `_DistilBertPipeline` that removes unsupported `token_type_ids` before calling the DistilBERT model. Use the project predictor instead of constructing a standard pipeline without this compatibility layer.

## Limitations

- The application classifies English text only.
- Predictions depend on the quality and domain of the GYAFC training data.
- Model artifacts are required locally and are not downloaded automatically.
- The repository does not include a separate automated test suite; the evaluation notebook is the primary model validation workflow.

## License and dataset

No license file is currently included in this repository. Review and add an appropriate project license before redistributing the code or trained artifacts. The training and evaluation data are based on the GYAFC formality corpus; follow the corpus terms and citation requirements when using or sharing the data.
