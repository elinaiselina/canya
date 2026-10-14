## CANYA a neural net to predict aggregation propensity
![Alt text](canyafig.png)
CANYA is a hybrid neural network that was trained on 100,000 random peptides to predict their aggregation status in a massively parallel experiment of aggregation rates. We include here the package and technical details for setting up and running CANYA on your own sequences. Please see [the preprint here](https://www.biorxiv.org/content/10.1101/2024.07.13.603366v2) for further information.

### Installation
To start, you'll need a python installation with tensorflow, numpy, and pandas installed. If you don't have these, CANYA will attempt to install the respective packages (and versions) with which it was developed. In this case, we recommend using a blank virtual environment or conda environment and installing CANYA from there.

e.g. by conda:
```
conda create -n canyaenv python=3.9
conda activate canyaenv
```

CANYA can then be installed via pip:
```
python -m pip install --no-cache-dir https://github.com/lehner-lab/canya/tarball/master
```

### Fitness regression in Python

On this branch, `create_model()` defaults to continuous fitness regression.
It preserves the original one-hot sequence, CNN, attention, and dense layers,
but uses a linear scalar output, mean squared error loss, and MAE/RMSE metrics.
Predictions can therefore be negative or positive.

```python
import numpy as np
from canya.modeling import create_model, get_embedded_seqs

# train_df and validation_df are separate, prepared data splits.
model = create_model(task="regression")
x_train = get_embedded_seqs(train_df["aa_seq"].tolist())
y_train = train_df["fitness_mean"].to_numpy(dtype=np.float32).reshape(-1, 1)
x_validation = get_embedded_seqs(validation_df["aa_seq"].tolist())
y_validation = validation_df["fitness_mean"].to_numpy(dtype=np.float32).reshape(-1, 1)

model.fit(
    x_train,
    y_train,
    validation_data=(x_validation, y_validation),
    batch_size=128,
    epochs=50,
)
metrics = model.evaluate(x_validation, y_validation, return_dict=True)
```

Use `fitness_mean` for the initial absolute assay-fitness baseline. When
evaluating unseen reference backgrounds, split by `sublibrary` before preparing
these arrays so variants of a reference stay in the same split. This model
requires training on continuous targets; the released classifier weights are
not trained fitness-regression weights.

To build the original classifier, use `create_model(task="classification")`.
`get_canya()` and the command-line interface continue to load the original
pretrained classifier and return aggregation scores.

### Pretrained aggregation predictions

Once installed, CANYA can be run very simply with the following options:

```--input``` Input sequences, either a FASTA or a text file with two *tab-delimited* columns with *no* header or column-names. Columns contain a sequence idenity (arbirtrary) as well as the amino acid sequence. See example data folder for examples.

```--output``` Name/directory of the output txt file. CANYA will output a single, tab-delimited file named after this prefix with two columns: (1) with the sequence identity (FASTA header or corresponding column of the input text file) (2) The CANYA nucleation score.

To run CANYA on the example file, run the following lines:
```
wget https://raw.githubusercontent.com/lehner-lab/canya/main/example_data/example.txt
canya --input example.txt --output example_out.txt
```

In addition, CANYA offers two other options:

```--summarize``` Either "no", which will report the CANYA score at every length-20 window along the sequence (rather than summarizing one score per sequence), or one of \{min, max, mean, median\}, which will summarize all scores along the sequence by using the specified function.

```--mode``` CANYA scores can be calculated using the model whose interpretations are presented in the paper (option "default"), or by taking the average of the top-10 most interpretable trained instances of CANYA ("ensemble"), which will also report the standard deviation of scores across the 10 models (i.e. epistemic uncertainty). 



CANYA has been tested on HPC and laptops---on a 2020 MacBook pro, using a single-core CPU, CANYA can generate predictions for roughly 100,000 sequences in less than a minute.


For help and other inqueries, please open a Github issue, or send an email to
```
mikejthomps(at)gmail(dot)com
```
