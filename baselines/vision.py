from sklearn import feature_extraction
import torch
from datasets import load_dataset, load_metric
from transformers import (
    ViTForImageClassification,
    ViTFeatureExtractor,
    AdamW,
    get_scheduler,
)
import logging
import math
from tqdm import tqdm
import numpy as np
import gc
from PIL import Image
import os
from copy import deepcopy


logger = logging.getLogger(__name__)


def run_train(args, **kwargs):
    accelerator = kwargs.get("accelerator")
    train_file = args.train_file
    valid_file = args.valid_file
    experiment = kwargs.get("experiment", None)

    data_files = dict()
    if train_file:
        data_files["train"] = train_file
    if valid_file:
        data_files["valid"] = valid_file

    if len(data_files) == 0:
        raise ValueError("Specify either train, valid, or test set")

    raw_datasets = load_dataset("csv", data_files=data_files)
    proc_datasets = raw_datasets

    feature_extractor = ViTFeatureExtractor.from_pretrained(args.vision_model)
    # raw_datasets["valid"] = raw_datasets["valid"].select(np.arange(20))
    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    # def preprocess_text(examples):
    #     inputs = tokenizer(
    #         examples["Text Transcription"],
    #         padding="max_length",
    #         max_length=args.max_seq_length,
    #         truncation=True,
    #     )

    #     batch = {**inputs, "labels": examples["misogynous"]}
    #     return batch

    # with accelerator.main_process_first():
    #     proc_datasets = raw_datasets.map(
    #         preprocess_text,
    #         batched=True,
    #         remove_columns=[
    #             #  "Text Transcription",
    #             "file_name",
    #             "shaming",
    #             "stereotype",
    #             "objectification",
    #             "violence",
    #         ],
    #         desc="Tokenizing text",
    #     )
    #     proc_datasets.set_format(
    #         "torch", columns=["input_ids", "attention_mask", "labels"]
    #     )
    #     logger.info(proc_datasets["train"][:2])

    def transform_data(examples):
        """Preprocess items on the fly at __getitem__ time"""
        images = [
            Image.open(os.path.join(args.data_dir, img_path)).convert("RGB")
            for img_path in examples["file_name"]
        ]

        vision_inputs = feature_extractor(
            images,
            return_tensors="pt",
        )

        labels = torch.LongTensor(examples["misogynous"])
        item = {**vision_inputs, "labels": labels}
        return item

    proc_datasets.set_transform(transform_data)

    #  Model
    model = ViTForImageClassification.from_pretrained(args.vision_model, num_labels=2)

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        proc_datasets["train"],
        shuffle=True,
        batch_size=args.per_device_train_batch_size,
        num_workers=args.preprocessing_num_workers,
        pin_memory=True,
    )

    if valid_file:
        eval_dataloader = torch.utils.data.DataLoader(
            proc_datasets["valid"],
            batch_size=args.per_device_eval_batch_size,
            num_workers=args.preprocessing_num_workers,
            pin_memory=True,
        )

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    if valid_file:
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader
        )
    else:
        model, optimizer, train_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader
        )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    # if accelerator.distributed_type == DistributedType.TPU:
    #     model.tie_weights()

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(
            args.max_train_steps / num_update_steps_per_epoch
        )

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(proc_datasets['train'])}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  LR scheduler = {args.lr_scheduler_type}")
    logger.info(f"  Total warmup steps = {args.num_warmup_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0
    best_val_loss = float("inf")
    early_stop_countdown = 0
    for epoch in range(args.num_train_epochs):

        model.train()
        for step, batch in enumerate(train_dataloader):
            output = model(**batch)
            loss = output.loss

            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if (
                step % args.gradient_accumulation_steps == 0
                or step == len(train_dataloader) - 1
            ):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

            if args.log_comet and (completed_steps % args.logging_steps == 0):
                experiment.log_metrics(
                    {"loss": loss.item()},
                    prefix="train",
                    step=completed_steps,
                    epoch=epoch,
                )

        if valid_file:
            metric = load_metric("f1")
            model.eval()
            losses = list()
            targets = list()
            for step, batch in tqdm(
                enumerate(eval_dataloader),
                desc="Validation",
                total=len(eval_dataloader),
            ):
                with torch.no_grad():
                    outputs = model(**batch)

                loss = outputs.loss
                losses.append(
                    accelerator.gather(loss.repeat(args.per_device_eval_batch_size))
                )

                predictions = outputs.logits.argmax(dim=-1)
                metric.add_batch(
                    predictions=accelerator.gather(predictions),
                    references=accelerator.gather(batch["labels"]),
                )
                targets.append(batch["labels"].detach().cpu())

            losses = torch.cat(losses)
            losses = losses[: len(proc_datasets["valid"])]
            try:
                mean_loss = torch.mean(losses)
            except OverflowError:
                mean_loss = float("inf")

            eval_metric = metric.compute()
            logger.info(f"epoch {epoch}: {eval_metric}, loss: {mean_loss.item()}")

            if args.log_comet:
                experiment.log_metrics(
                    {"loss": mean_loss.item(), "f1": eval_metric["f1"]},
                    epoch=epoch,
                    step=completed_steps,
                    prefix="valid",
                )

            if mean_loss < best_val_loss:
                logger.info(f"Improved loss from {best_val_loss} to {mean_loss}!")
                best_val_loss = mean_loss
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    args.output_dir, save_function=accelerator.save
                )
                early_stop_countdown = 0
            else:
                if args.early_stop_patience > 0:
                    early_stop_countdown += 1
                    logger.info(
                        f"Early stop countdown: {early_stop_countdown}/{args.early_stop_patience}"
                    )

    if args.output_dir is not None and args.save_last:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            f"{args.output_dir}/last", save_function=accelerator.save
        )

    results = None
    if valid_file:
        # get final predictions on the valid set
        logger.info("Running inference on the valid set")
        args.model_name_or_path = args.output_dir
        results = run_test(
            args, accelerator=accelerator, test_dataset=raw_datasets["valid"]
        )
        valid = load_dataset("csv", data_files={"valid": valid_file})["valid"]
        results["targets"] = torch.LongTensor(valid["misogynous"])

    accelerator.free_memory()

    logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated()}")
    logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved()}")

    return results


def run_test(args, **kwargs):
    accelerator = kwargs.get("accelerator")
    test_dataset = kwargs.get("test_dataset")
    feature_extractor = ViTFeatureExtractor.from_pretrained(args.vision_model)

    def transform_data(examples):
        """Preprocess items on the fly at __getitem__ time"""
        images = [
            Image.open(os.path.join(args.data_dir, img_path)).convert("RGB")
            for img_path in examples["file_name"]
        ]

        vision_inputs = feature_extractor(
            images,
            return_tensors="pt",
        )

        labels = torch.LongTensor(examples["misogynous"])
        item = {**vision_inputs, "labels": labels}
        return item

    test_dataset.set_transform(transform_data)

    #  Model
    model = ViTForImageClassification.from_pretrained(args.model_name_or_path)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.preprocessing_num_workers,
        pin_memory=True,
    )

    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    model.eval()
    scores = list()
    preds = list()

    for _, batch in tqdm(
        enumerate(test_dataloader), desc="Testing", total=len(test_dataloader)
    ):
        with torch.no_grad():
            outputs = model(**batch)

        scores.append(outputs.logits.softmax(dim=-1))
        predictions = outputs.logits.argmax(dim=-1)
        preds.append(predictions)

    scores = torch.cat(scores).detach().cpu()
    preds = torch.cat(preds).detach().cpu()

    return {
        "scores": scores,
        "preds": preds,
    }
