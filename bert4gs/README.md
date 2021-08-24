# BERT for GrandStanding


We train a BERTForSequenceClassification Model to estimate the amount of grandstanding in judicial speech. The labelled data is from [When Do Politicians Grandstand](https://www.journals.uchicago.edu/doi/abs/10.1086/709147?af=R). We use the [hugginface](https://huggingface.co/) implementation of BERT to perform fine-tuning and inference. 

We initialize the bert sequence classification model with the `bert-base-uncased` weights perform all training and evaluation in `train_b4gs.py`. In `infer_gs.py` we load the model and infer on segments of judicial speech extracted from transcriptions that are prepared in the `S2T_API` folder. The `batch_infer` script performs the `infer_gs.py` processing and cleaning across the permutations of diarization and speech-to-text methods.

This training and inference was done in colab notebooks for GPU usage, if there are any issues other than filepaths for running locally or on a cluster please let me know. 