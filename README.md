# Hierarchical text categorization of South African research outputs #
This objective of this project is to build a hierachical text classifier for South African research outputs. We consider different feature extraction methods to obtain a semantic representation of the text document which comprises the title and abstract of a research output. We evalaute several classifier architectures which include Convolutional Neural Networks, Recurrent Neural networks and attention mechanisms.

## Install dependencies ##
    $pip install -r requirements.txt

## Dataset construction and feature extraction ##
### Data.py ###
    Builds the dataset, removes invalid entries and balances dataset:
    $python Data.py
### LDA.py ###
    Obtains embeddings from LDA model:
    $python LDA.py

### FeatureExtraction.py ###
    Obtain embeddings from SciBERT model and performs fusion strategies:
    $python FeatureExtraction.py

## Model training and evaluation ##
### ClassifierTuning.py ###
    Trains, tunes and evaluates four of the classifier architectures for the fusion strategies.
    $python ClassifierTuning.py

### AttentionEmbedding.py ###
    Trains, tunes and evaluates label embedding CNN for the fusion strategies.
    $python ClassifierTuning.py

### TokenizerClassifier.py ###
    Trains, tunes and evaluates four of the classifier architectures for baseline feature extraction method.
    $python TokenizerClassifier.py

### TokenizerEmbeddingAttention.py ###
    Trains, tunes and evaluates label embedding CNN for baseline feature extraction method.
    $python TokenizerEmbeddingAttention.py