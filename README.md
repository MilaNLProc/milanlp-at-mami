# MilaNLP at SemEval-2022 Task 5: Using Perceiver IO for Detecting Misogynous Memes with Text and Image Modalities

**Paper**

[https://aclanthology.org/2022.semeval-1.90/](https://aclanthology.org/2022.semeval-1.90/)

We introduce the system proposed by the MilaNLP team for the Multimedia Automatic Misogyny Identification (MAMI) challenge. We use Perceiver IO as a multimodal late fusion over unimodal streams to address both sub-tasks A and B. We build unimodal embeddings using Vision Transformer (image) and RoBERTa (text transcript). We enrich the input representation using face and demographic recognition, image captioning, and detection of adult content and web entities. To the best of our knowledge, this work is the first to use Perceiver IO combining text and image modalities. The proposed approach outperforms unimodal and multimodal baselines.

```bibtex
@inproceedings{attanasio-etal-2022-milanlp,
    title = "{M}ila{NLP} at {S}em{E}val-2022 Task 5: Using Perceiver {IO} for Detecting Misogynous Memes with Text and Image Modalities",
    author = "Attanasio, Giuseppe  and
      Nozza, Debora  and
      Bianchi, Federico",
    booktitle = "Proceedings of the 16th International Workshop on Semantic Evaluation (SemEval-2022)",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.semeval-1.90",
    pages = "654--662",
}
```

## Getting started

### Environment

We use Python 3.6+. You need to have a fully functional pytorch environment. File `requirements.txt` contains the project dependencies.

### Data

We are not allowed to distribute the shared task dataset. Please refer to the task organizers to get access to it.

In this repository, under `data`, you will pre-computed information to build unimodal streams for both training and testing memes:

- fairface.tsv: faces and demographics detected with FairFace;
- image_captions.tsv: captions generated with Show, Attend and Tell;
- nsfw.tsv: information on whether adult content is found with NudeNet;
- web_entities.tsv: results from Google Cloud Vision APIs queried with meme images.  

## Train our system

### Cross validation

Run cross validation using unimodal streams and multimodal Perceiver IO as late fusion:

    python runner.py \
        --do_cv \
        --model_type perceiver \
        --text_model roberta-base \
        --output_dir results/cv/vit-roberta-perceiver \
        --vision_model google/vit-base-patch16-224-in21k \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 8 \
        --weight_decay 0.01 \
        --num_train_epochs 5 \
        --num_warmup_steps 300 \
        --learning_rate 1e-5 \
        --seed 42 \
        --max_seq_length 32 \
        --preprocessing_num_workers 4 \
        --device cuda \
        --save_last \
        --num_latents 256 \
        --latent_dim 512 \
        --depth 6


Run cross validation using unimodal RoBERTa:

    python runner.py \
        --do_cv \
        --output_dir ./results/cv/roberta-large \
        --model_type text \
        --model_name_or_path roberta-large \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --weight_decay 0.01 \
        --num_train_epochs 5 \
        --num_warmup_steps 20 \
        --seed 42 \
        --max_seq_length 64 \
        --preprocessing_num_workers 4 \
        --early_stop_patience 3 \
        --save_best


Run cross validation using unimodal Vision Transformer (ViT):

    python runner.py \
        --do_cv \
        --model_type vision \
        --data_dir ./data/images \
        --output_dir results/task_A/cv/vit \
        --vision_model google/vit-base-patch16-224-in21k \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 8 \
        --weight_decay 0.01 \
        --num_train_epochs 4 \
        --num_warmup_steps 200 \
        --learning_rate 1e-5 \
        --seed 42 \
        --preprocessing_num_workers 4 \
        --device cuda \
        --save_last \
        --taskA 

### Final training

Run training on the entire training dataset

    python runner.py \
        --do_train \
        --train_file ./data/training/training.csv \
        --model_type perceiver \
        --output_dir ./results/task_B/vit-roberta-perceiver-augment \
        --text_model roberta-base \
        --vision_model google/vit-base-patch16-224-in21k \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 8 \
        --weight_decay 0.01 \
        --num_train_epochs 3 \
        --num_warmup_steps 200 \
        --learning_rate 1e-5 \
        --seed 42 \
        --max_seq_length 32 \
        --preprocessing_num_workers 4 \
        --save_last \
        --device cuda \
        --depth 6 \
        --num_latents 128 \
        --latent_dim 512

