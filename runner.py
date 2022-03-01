import argparse
from comet_ml import Experiment
import os
import logging
import math
import os
import IPython
import pdb
import pandas as pd
import json

import datasets
import torch
from tqdm.auto import tqdm
from traitlets.traitlets import default
from torchvision.io import read_image, ImageReadMode
import numpy as np
import transformers
from accelerate import Accelerator
from transformers import (
    set_seed,
)

from baselines import text, vision, mm
from utils import compute_metrics, TEST_FILE, read_test_dataset


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model")
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--do_cv", action="store_true")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_test", action="store_true")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--train_file")
    parser.add_argument("--valid_file")
    parser.add_argument("--text_model")
    parser.add_argument("--vision_model")
    # parser.add_argument(
    #     "--config_name",
    #     type=str,
    #     default=None,
    #     help="Pretrained config name or path if not the same as model_name",
    # )
    # parser.add_argument(
    #     "--pad_to_max_length",
    #     action="store_true",
    #     help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    # )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=5,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        # type=SchedulerType,
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=float,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=2,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache",
        type=bool,
        default=False,
        help="Overwrite the cached training and evaluation sets",
    )

    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--log_comet", action="store_true")
    parser.add_argument("--comet_tags", nargs="*")
    parser.add_argument("--data_dir", type=str, default=None)

    # loss type
    parser.add_argument("--loss_type", type=str, default="focal")
    parser.add_argument("--weighted_loss", action="store_true")

    # save strategy
    parser.add_argument("--save_best", action="store_true")
    parser.add_argument("--save_last", action="store_true")

    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--taskA", action="store_true")

    # perceiver
    parser.add_argument("--num_latents", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--add_pos_emb", action="store_true")
    parser.add_argument("--weight_tie", action="store_true")
    parser.add_argument("--use_separate_projs", action="store_true")

    # additional features
    parser.add_argument("--add_features", action="store_true")

    # caption
    parser.add_argument("--add_caption", action="store_true")

    # text augmentation
    parser.add_argument("--text_augmentation", action="store_true")
    parser.add_argument("--text_mask_prob", type=float, default=0.2)

    # freeze backbone encoders
    parser.add_argument("--num_epochs_frozen_encoders", type=int, default=0)

    # add web entities
    parser.add_argument("--add_web_entities", action="store_true")

    args = parser.parse_args()

    if not args.tokenizer_name:
        args.tokenizer_name = args.model_name_or_path

    return args


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.seed is not None:
        set_seed(args.seed)

    accelerator = Accelerator()

    if args.log_comet:
        # Create an experiment with your api key
        experiment = Experiment(
            api_key=os.getenv("COMET_API_KEY"),
            project_name="hateful-memes",
            workspace="g8a9",
        )
        experiment.log_parameters(args)
        if args.comet_tags:
            experiment.add_tags(args.comet_tags)
    else:
        experiment = None

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)
    logger.setLevel(
        logging.INFO if accelerator.is_local_main_process else logging.ERROR
    )
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # if accelerator.is_main_process:
    #     os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    if args.do_cv:
        targets = list()
        preds = list()

        targets = list()
        pbar = tqdm(total=3, desc="CV split")
        for i in range(3):
            args.train_file = os.path.join("data", "training", f"cv_{i}", "train.csv")
            args.valid_file = os.path.join("data", "training", f"cv_{i}", "valid.csv")

            if args.model_type == "text":
                output = text.run_train(
                    args,
                    accelerator=accelerator,
                    experiment=experiment,
                )
            elif args.model_type == "vision":
                output = vision.run_train(
                    args,
                    accelerator=accelerator,
                    experiment=experiment,
                )
            elif args.model_type == "perceiver":
                output = mm.run_train(
                    args,
                    accelerator=accelerator,
                    experiment=experiment,
                )
            else:
                raise NotImplementedError()

            targets.append(output["targets"])
            preds.append(output["preds"])

            pbar.update(1)

        pbar.close()

        targets = torch.cat(targets)
        preds = torch.cat(preds)

        metrics = compute_metrics(targets.numpy(), preds.numpy())

        print()
        print("##### CV RESULTS #####")
        print(f"F1 Macro {metrics['f1_macro']}")
        print(metrics)

        if args.log_comet:
            experiment.log_metrics(metrics)

        #  save results
        # torch.save(output["scores"], os.path.join(args.output_dir, "scores.pt"))
        res_file = os.path.join(args.output_dir, "results.json")
        with open(res_file, "w") as fp:
            json.dump(metrics, fp)

    elif args.do_train:
        if args.model_type == "text":
            output = text.run_train(
                args, accelerator=accelerator, experiment=experiment
            )
        elif args.model_type == "perceiver":
            output = mm.run_train(args, accelerator=accelerator, experiment=experiment)
        else:
            raise NotImplementedError()

    elif args.do_test:
        test_dataset = read_test_dataset()

        if args.model_type == "text":
            output = text.run_test(
                args,
                accelerator=accelerator,
                experiment=experiment,
                test_dataset=test_dataset,
            )
        elif args.model_type == "perceiver":
            output = mm.run_test(
                args,
                accelerator=accelerator,
                experiment=experiment,
                test_dataset=test_dataset,
            )
        else:
            raise NotImplementedError()

        test_df = pd.read_csv(TEST_FILE, sep="\t")

        if output["preds"].ndim == 2:  # multi task predictions
            test_df = pd.concat(
                [test_df, pd.DataFrame(output["preds"].numpy())], axis=1
            )
        else:
            test_df["preds"] = output["preds"].numpy()

        test_df = test_df.drop(columns=["Text Transcription"])
        test_df.to_csv(
            os.path.join(args.output_dir, "answer.txt"),
            sep="\t",
            header=False,
            index=None,
        )

    else:
        raise ValueError("Specify do_cv, do_train, or do_test")


if __name__ == "__main__":
    main()