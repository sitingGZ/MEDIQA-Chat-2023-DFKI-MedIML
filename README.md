# MEDIQA-Chat-2023-DFKI-MedIML
Repo for MEDIQA-Chat-2023 TaskA

[Google drive link](https://drive.google.com/drive/folders/1soKGJLSmqZAvXK1fNTYJpS2U1xaKRANd?usp=sharing) for obtaining the fine-tuned checkpoints for task A (predict_header.zip and summarization.zip).

1. To regenerate the results: unzip the predict_header.zip and summarization.zip and place them in the directory of `ckpts` under this main directory. 
2. At the moment, for initializing the base biogpt model from microsoft/biogpt, it would take a while to downloaded the model to initialize the chat2note model in the train_biogpt_sumpy and inference.py from `biogpt_modules`. 
3. Inference time on cpu is around 3 hours and on gpu (32GB) about 30 minutes. 

