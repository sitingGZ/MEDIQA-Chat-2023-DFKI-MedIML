# MEDIQA-Chat-2023-DFKI-MedIML
Repo for MEDIQA-Chat-2023 TaskA

[Google drive link](https://drive.google.com/drive/folders/1soKGJLSmqZAvXK1fNTYJpS2U1xaKRANd?usp=sharing) for obtaining the fine-tuned checkpoints for task A (predict_header.zip and summarization.zip, extract and remove "001" or "002" suffix in the name of directory that assigned during downloading").
  - Approach: fine-tunig biogpt_base for TaskA. The fine-tuned model is trained to handle list of context input that are parts of one dialogue and can generate target section header as well as summary of one dialogue.
  - To regenerate the results: unzip the predict_header.zip and summarization.zip and place them in the directory of `ckpts` under this main directory. 
  - At the moment, the model is initialized with the model config from microsoft/biogpt, which is the base model for the chat2note model to be trained with train_biogpt_sum.py and obtain results in inference.py from `biogpt_modules`. Loading the weights from fine-tuned checkpoints adapts the base model for TaskA.
  - Inference time on cpu is around 2-3 hours and on gpu (32GB) about 30 minutes. 
  - Train, valid, and test csv files are saved in `TaskA` directory, so running the decode scripts directly can get the results of each run. 

