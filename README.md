# MEDIQA-Chat-2023-DFKI-MedIML
Repo for MEDIQA-Chat-2023 TaskA

[Google drive link](https://drive.google.com/drive/folders/1soKGJLSmqZAvXK1fNTYJpS2U1xaKRANd?usp=sharing) for obtaining the fine-tuned checkpoints for task A (predict_header.zip and summarization.zip).

1. Unzip the predict_header.zip and summarization.zip and place them in the directory of `ckpts`. 
2. Generated results of three runs are saved in `results` directory.
3. At the moment, initializing the base biogpt model from microsoft/biogpt would be downloaded to initialize the chat2note model in the train_biogpt_sumpy and inference.py from `biogpt_modules`. 

