# BERT for uncertainty detection
Bachelor's thesis available via this [link](https://www.hse.ru/en/edu/vkr/366007531).

![](https://github.com/PeterZhizhin/BERTUncertaintyDetection/raw/master/pictures/BERT_demo.png)

## Project description

### Dataset

The models here are trained to detect uncertain words and sentences in text and the class of uncertainty:
1. Dynamic. Indicates necessity, dispositions, external circumstances, wishes, intents, plans,
and desires. Example: I **have to** go.
2. Doxastic. Expresses the speaker’s beliefs. Example: He **believes** that the Earth is flat.
3. Investigation. Propositions, for which the truth value cannot be stated until further
analysis is done. Example: We **examined** the role of NF-kappa B in protein activation.
4. Conditional. Used for conditionals. Example: **If** it rains, we’**ll** stay in.
5. Epistemic. Uncertainty, for which it is known that the proposition is neither true nor
false. Example: It **may** be raining.

The dataset used here is the re-annotated CoNLL-2010 shared task dataset for uncertainty detection
(Szeged Uncertainty Corpus). It's available for download [here](https://rgai.sed.hu/file/139).
The dataset consists of two main parts:
1. Wikipedia (WikiWeasel).
2. Biological (BioScope).

### Goal

The goal of this project was to compare the performance of two different BERT training procedures:
1. Train a domain-specific model: [SciBERT](https://github.com/allenai/scibert) and [BioBERT](https://github.com/dmis-lab/biobert) on Biological part of the dataset.
2. Train the general-domain [BERT](https://github.com/google-research/bert) on the Wikipedia part and perform transfer-learning.

20 models with different seeds are trained and F1-score is compared with statistical tests.

## Results

I showed that it's not possible to conclude on this dataset which approach yields better results.
But most importantly, SciBERT almost always outperforms BioBERT and doesn't benefit as much from additional Wikipedia data.
If you decide to train a domain-specific language model, train it from a random initialization with a domain-specific
dictionary rather that start with BERT initialization.

## Model

SciBERT model for uncertainty detection on biological texts is available on my [Google Drive](https://drive.google.com/file/d/1rnXWW69HSf31ouRLydZ9jRzGGtIZFZlA/view?usp=sharing).

## Demo

The demo allows to experiment with the model and annotate an arbitrary text for uncertainty.

Instructions to run the demo:
* Clone the repository:
```
git clone https://github.com/PeterZhizhin/BERTUncertaintyDetection
```
* Create a virtualenv and install dependencies:
```
python -m venv .env
source .env/bin/activate
pip install -U spacy
python -m spacy download en_core_web_sm
pip install aiohttp jinja2 aiohttp-jinja2 transformers torch torchvision
```
* Go to the demo folder:
```
cd demo
```
* Download the [model](https://github.com/PeterZhizhin/BERTUncertaintyDetection) extract the archive remember the path to the model folder.
* Run the server:
```
python demo_server.py --model_path [PATH TO FOLDER WITH THE MODEL] --labels_path ../labels.txt
```

## Train the models yourself

All the model training was done on a Slurm cluster of National Research University Higher School of Economics.
So, all the scripts for training require a Slurm cluster with GPUs by default.
If you wish to train models without a Slurm cluster, you may change the training scripts.

* Clone the repo:
```
git clone https://github.com/PeterZhizhin/BERTUncertaintyDetection
```
* Install dependencies:
```
python -m venv .env
source .env/bin/activate
pip install -U spacy
python -m spacy download en_core_web_sm
pip install aiohttp jinja2 aiohttp-jinja2 transformers torch torchvision lxml
```
* [Download](https://rgai.sed.hu/file/139) the dataset, extract it and place to the `uncertainty_dataset` folder.
* Make all shell scripts executable:
```
chmod +x *.sh
chmod +x huggingface_models/*.sh
```
* Create the datasets for training and evaluation:
```
./create_biomedical_ner_dataset_train_test.sh
./create_biomedical_classification_dataset_train_test.sh
./create_wiki_classification_dataset_train_test.sh
./create_wiki_ner_dataset_train_test.sh
```
* Train all models:
```
cd huggingface_models
sbatch --wait ./train_all_models_on_wiki_and_bio_slurm.sh
sbatch --wait ./transfer_all_models_on_wiki_and_bio_slurm.sh
sbatch --wait ./train_all_classification_models_on_wiki_and_bio_slurm.sh
sbatch --wait ./transfer_all_classification_models_on_wiki_and_bio_slurm.sh
```
* All models are now available in `ner_experiments` and `classification_experiment`.