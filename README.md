# MEDIQA-Chat-2023-DFKI-MedIML
Repo for MEDIQA-Chat-2023 TaskA

[Google drive link](https://drive.google.com/drive/folders/1soKGJLSmqZAvXK1fNTYJpS2U1xaKRANd?usp=sharing) for obtaining the fine-tuned checkpoints for task A (predict_header.zip and summarization.zip, extract and remove "001" or "002" suffix in the name of directory that assigned during downloading").
  - Approach: fine-tunig biogpt_base for TaskA. The fine-tuned model is trained to handle list of context input that are parts of one dialogue and can generate target section header as well as summary of one dialogue.
  - To regenerate the results: unzip the predict_header.zip and summarization.zip and place them in the directory of `ckpts` under this main directory. 
  - At the moment, for initializing the base biogpt model from microsoft/biogpt, it would take a while to downloaded the model to initialize the chat2note model in the train_biogpt_sumpy and inference.py from `biogpt_modules`. Loading the weights from fine-tuned checkpoints adapts the base model for TaskA.
  - Inference time on cpu is around 3 hours and on gpu (32GB) about 30 minutes. 
  - TaskA train, valid, and test csv files are saved in `TaskA` directory, so run decode scripts directly can get the results of each run. 

