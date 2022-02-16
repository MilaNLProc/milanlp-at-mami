from transformers import (
    AutoTokenizer,
    AutoFeatureExtractor,
    AdamW,
    get_scheduler,
)
from datasets import load_dataset, load_metric
import logging
from PIL import Image
import torch
from baselines.perceiver import PerceiverPooler
import math
from tqdm import tqdm
from einops import rearrange
from torchvision.ops.focal_loss import sigmoid_focal_loss
from kornia.losses import focal_loss
from utils import compute_metrics, compute_metrics_multi_task
import os
from pprint import pprint
from utils import read_test_dataset
from sklearn.preprocessing import LabelEncoder
from torchvision.transforms import *
import pandas as pd
from functools import partial


logger = logging.getLogger(__name__)


def prepare_targets(batch, multi_task=False):
    if multi_task:
        targets = torch.stack(
            (
                batch["misogynous"],
                batch["shaming"],
                batch["stereotype"],
                batch["objectification"],
                batch["violence"],
            ),
            axis=-1,
        )  # b x n_tasks
    else:
        targets = rearrange(batch["misogynous"], "b -> b ()")

    return targets


def _transform():
    """
    Data Augmentation
    """
    return Compose(
        [
            ColorJitter(hue=0.1),
            RandomHorizontalFlip(),
            RandomAffine(
                degrees=15,
                translate=(0.1, 0.1),
                scale=(0.8, 1.2),
                shear=(-15, 15, -15, 15),
                interpolation=InterpolationMode.BILINEAR,
                fill=127,
            ),
            RandomPerspective(
                distortion_scale=0.3,
                p=0.3,
                interpolation=InterpolationMode.BILINEAR,
                fill=127,
            ),
            RandomAutocontrast(p=0.3),
            RandomEqualize(p=0.3),
        ]
    )


VISION_FEATURES = 196
TEXT_FEATURES = 32
ADDITIONAL_FEATURES = 64
CAPTION_FEATURES = 32
ENTITIES_FEATURES = 32


def prepare_dataset(args, raw_datasets, include_targets=True, multi_task=False):
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.vision_model)
    le = None
    if args.add_features:
        nsfw = pd.read_csv("data/nsfw.tsv", sep="\t", index_col="file_name")
        fairface = pd.read_csv("data/fairface.tsv", sep="\t").set_index("file_name")
        le = LabelEncoder().fit(
            fairface.age.tolist()
            + fairface.gender.tolist()
            + fairface.race.tolist()
            + nsfw.is_safe_bool.astype(str).tolist()
            + list(range(21))
            + ["[PAD]"]
        )

    if args.add_caption:
        captions = pd.read_csv("data/image_captions.tsv", sep="\t", index_col="image")
        cap_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    if args.add_web_entities:
        web_entities = pd.read_csv(
            "data/web_entities.tsv", sep="\t", index_col="file_name"
        ).fillna("")
        ent_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def compute_additional_features(file_names):
        features = list()
        feature_mask = list()
        for file_name in file_names:
            f = list()
            m = list()

            nsfw_id = nsfw.loc[file_name]["is_safe_bool"].astype(str)
            f.append(nsfw_id)
            m.append(1)

            try:
                ff = fairface.loc[file_name]

                if ff.ndim == 1:  # single face found
                    f.extend([ff.age, ff.race, ff.gender])
                    m.extend([1, 1, 1])
                else:
                    for row in ff.iterrows():  # multiple faces
                        f.extend([row.age, row.race, row.gender])
                        m.extend([1, 1, 1])

                f.append(ff.shape[0])
                m.append(1)
            except:
                count = 0
                f.append(0)  # last index is the face counts
                m.append(1)

            # Padding
            to_pad = ADDITIONAL_FEATURES - len(m)
            if to_pad > 0:
                m.extend([0] * to_pad)
                f.extend(["[PAD]"] * to_pad)

            assert len(m) == ADDITIONAL_FEATURES
            assert len(f) == ADDITIONAL_FEATURES

            # transform to indexes
            features.append(le.transform(f))
            feature_mask.append(m)

        return features, feature_mask

    def compute_caption(file_names):
        texts = captions.loc[file_names]["caption"].tolist()
        cap_inputs = cap_tokenizer(
            texts,
            padding="max_length",
            max_length=CAPTION_FEATURES,
            truncation=True,
        )
        return cap_inputs["input_ids"], cap_inputs["attention_mask"]

    def compute_web_entities(file_names):
        texts = web_entities.loc[file_names].values.tolist()
        ent_inputs = ent_tokenizer(
            texts,
            padding="max_length",
            max_length=ENTITIES_FEATURES,
            truncation=True,
        )

        return ent_inputs["input_ids"], ent_inputs["attention_mask"]

    def preprocessing(examples):
        text_inputs = tokenizer(
            examples["Text Transcription"],
            padding="max_length",
            max_length=args.max_seq_length,
            truncation=True,
        )
        paths = [f"data/images/{f}" for f in examples["file_name"]]

        item = {"img": paths, **text_inputs}

        if args.add_features:
            f, fm = compute_additional_features(examples["file_name"])
            item.update({"features": f, "feature_mask": fm})

        if args.add_caption:
            c, cm = compute_caption(examples["file_name"])
            item.update({"caption": c, "caption_mask": cm})

        if args.add_web_entities:
            e, em = compute_web_entities(examples["file_name"])
            item.update({"entities": e, "entities_mask": em})

        return item

    # tokenize text
    proc_datasets = raw_datasets.map(
        preprocessing,
        batched=True,
        remove_columns=["file_name"],
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    def transform_data(examples):
        """Preprocess items on the fly at __getitem__ time"""

        # image augmentation
        augmenter = _transform()

        images = [
            augmenter(Image.open(img_path).convert("RGB"))
            for img_path in examples["img"]
        ]

        vision_inputs = feature_extractor(
            images,
            return_tensors="pt",
        )

        # text augmentation
        attention_mask = torch.tensor(examples["attention_mask"])
        if args.text_augmentation:
            prob_mask = torch.rand(attention_mask.shape)
            attention_mask[prob_mask < args.text_mask_prob] = 0

        item = {
            **vision_inputs,
            "input_ids": torch.tensor(examples["input_ids"]),
            "attention_mask": attention_mask,
        }

        # build additional features
        _include = list()
        if args.add_features:
            _include.extend(["features", "feature_mask"])
        if args.add_caption:
            _include.extend(["caption", "caption_mask"])
        if args.add_web_entities:
            _include.extend(["entities", "entities_mask"])
        if _include:
            item.update({k: torch.tensor(examples[k]) for k in _include})

        # define targets
        if include_targets:
            if multi_task:
                item.update(
                    {
                        "misogynous": torch.LongTensor(examples["misogynous"]),
                        "shaming": torch.LongTensor(examples["shaming"]),
                        "stereotype": torch.LongTensor(examples["stereotype"]),
                        "objectification": torch.LongTensor(
                            examples["objectification"]
                        ),
                        "violence": torch.LongTensor(examples["violence"]),
                    }
                )
            else:
                item["misogynous"] = torch.LongTensor(examples["misogynous"])

        return item

    proc_datasets.set_transform(transform_data)

    return proc_datasets, le


def get_input_length(args):
    input_seq_len = VISION_FEATURES + TEXT_FEATURES
    if args.add_features:
        input_seq_len += ADDITIONAL_FEATURES

    if args.add_caption:
        input_seq_len += CAPTION_FEATURES
        logger.info(f"Adding caption of max length {CAPTION_FEATURES}")

    if args.add_web_entities:
        input_seq_len += ENTITIES_FEATURES
        logger.info(f"Adding web entities of max length {ENTITIES_FEATURES}")

    return input_seq_len


def loss_fct(args):
    if args.loss_type == "focal":
        return partial(focal_loss, alpha=0.25, gamma=2, reduction="mean")
    else:
        return torch.nn.CrossEntropyLoss()


def run_train(args, **kwargs):
    accelerator = kwargs.get("accelerator", None)
    device = args.device
    multi_task = not args.taskA
    experiment = kwargs.get("experiment", None)

    data_files = dict()
    data_files["train"] = args.train_file
    if args.valid_file:
        data_files["valid"] = args.valid_file

    if len(data_files) == 0:
        raise ValueError("Specify either train, valid, or test set")

    raw_datasets = load_dataset("csv", data_files=data_files)
    logger.info(raw_datasets["train"][0])
    # raw_datasets["valid"] = raw_datasets["valid"].select(np.arange(20))

    proc_datasets, le = prepare_dataset(args, raw_datasets, multi_task=multi_task)

    # the final length of the input array, concatenating all modalities
    input_seq_len = get_input_length(args)
    if args.add_features:
        feature_space = le.classes_.size

    model = PerceiverPooler(
        text_model=args.text_model,
        vision_model=args.vision_model,
        n_tasks=5 if multi_task else 1,
        n_classes=2,
        input_size=768,
        input_seq_len=input_seq_len,
        add_pos_emb=args.add_pos_emb,
        add_feature_space=feature_space if args.add_features else None,
        add_caption=args.add_caption,
        add_web_entities=args.add_web_entities,
        use_separate_projs=args.use_separate_projs,
        depth=args.depth,
        num_latents=args.num_latents,
        latent_dim=args.latent_dim,
        cross_heads=1,
        latent_heads=8,
        cross_dim_head=64,
        latent_dim_head=64,
        weight_tie_layers=False,
        decoder_ff=True,
    )
    model = model.to(device)
    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        proc_datasets["train"],
        shuffle=True,
        batch_size=args.per_device_train_batch_size,
        num_workers=args.preprocessing_num_workers,
        pin_memory=True,
    )

    if args.valid_file:
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
    # optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # if valid_file:
    #     model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    #         model, optimizer, train_dataloader, eval_dataloader
    #     )
    # else:
    #     model, optimizer, train_dataloader = accelerator.prepare(
    #         model, optimizer, train_dataloader
    #     )

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

    # metric = load_metric("f1")

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        # * accelerator.num_processes
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
    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0
    best_val_loss = float("inf")
    early_stop_countdown = 0
    for epoch in range(args.num_train_epochs):

        # freeze / unfreeze text and vision encoders
        if epoch < args.num_epochs_frozen_encoders and not model.has_frozen_encoders:
            logger.info("Freezing backbone encoders!")
            model.freeze_encoders()
        if epoch >= args.num_epochs_frozen_encoders and model.has_frozen_encoders:
            model.unfreeze_encoders()
            logger.info("Unfreezing backbone encoders!")

        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            logits = model(**batch)
            logits = rearrange(logits, "b t c -> b c t")  # b x 2 x 5
            targets = prepare_targets(batch, multi_task=multi_task)
            loss = focal_loss(logits, targets, alpha=0.25, gamma=2, reduction="mean")

            loss = loss / args.gradient_accumulation_steps
            # accelerator.backward(loss)
            loss.backward()

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

        if args.valid_file:

            model.eval()
            losses = list()
            predictions = list()
            gold = list()
            for step, batch in tqdm(
                enumerate(eval_dataloader),
                desc="Validation",
                total=len(eval_dataloader),
            ):
                batch = {k: v.to(device) for k, v in batch.items()}

                with torch.no_grad():
                    logits = model(**batch)

                logits = rearrange(logits, "b t c -> b c t")  # b x 2 x n_tasks
                targets = prepare_targets(batch, multi_task=multi_task)  # b x n_tasks

                loss = loss_fct(args)(logits, targets)
                losses.append(loss.item())

                predictions.append(logits.argmax(dim=1).detach().cpu())
                gold.append(targets.cpu())
                # metric.add_batch(
                #     predictions=accelerator.gather(predictions),
                #     references=accelerator.gather(batch["labels"]),
                # )

            # losses = torch.cat(losses)
            # losses = losses[: len(proc_datasets["valid"])]

            predictions = torch.cat(predictions)
            gold = torch.cat(gold)
            try:
                mean_loss = torch.mean(torch.tensor(losses))
            except OverflowError:
                mean_loss = float("inf")

            # eval_metric = metric.compute()
            # logger.info(f"epoch {epoch}: {eval_metric}, loss: {mean_loss.item()}")

            if multi_task:
                metrics = compute_metrics_multi_task(
                    predictions.numpy(),
                    gold.numpy(),
                    names=[
                        "misogynous",
                        "shaming",
                        "stereotype",
                        "objectification",
                        "violence",
                    ],
                )
            else:
                metrics = compute_metrics(predictions.numpy(), gold.numpy())

            logger.info("#" * 10)
            logger.info("Validation metrics")
            logger.info(f"{metrics}")
            pprint(metrics)

            if args.log_comet:

                if multi_task:
                    to_log = {f"f1_{k}": v["f1_macro"] for k, v in metrics.items()}
                else:
                    to_log = metrics

                experiment.log_metrics(
                    {"loss": mean_loss.item(), **to_log},
                    epoch=epoch,
                    step=completed_steps,
                    prefix="valid",
                )

            if mean_loss < best_val_loss:
                logger.info(f"Improved loss from {best_val_loss} to {mean_loss}!")
                best_val_loss = mean_loss

                if args.output_dir is not None and args.save_best:
                    torch.save(
                        model.state_dict(),
                        os.path.join(args.output_dir, "weights_best.pt"),
                    )
                # accelerator.wait_for_everyone()
                # unwrapped_model = accelerator.unwrap_model(model)
                # unwrapped_model.save_pretrained(
                #     args.output_dir, save_function=accelerator.save
                # )
                early_stop_countdown = 0
            else:
                if args.early_stop_patience > 0:
                    early_stop_countdown += 1
                    logger.info(
                        f"Early stop countdown: {early_stop_countdown}/{args.early_stop_patience}"
                    )

    if args.output_dir is not None and args.save_last:
        torch.save(model.state_dict(), os.path.join(args.output_dir, "weights_last.pt"))

    #  Final validation with best/last model
    results = None
    if args.valid_file:
        logger.info("Running inference on the valid set")

        if args.save_best:
            args.model_name_or_path = os.path.join(args.output_dir, "weights_best.pt")
        else:
            args.model_name_or_path = os.path.join(args.output_dir, "weights_last.pt")

        results = run_test(
            args, accelerator=accelerator, test_dataset=raw_datasets["valid"]
        )

        if multi_task:
            raise NotImplementedError()
        else:
            targets = torch.LongTensor(raw_datasets["valid"]["misogynous"])
        results["targets"] = targets

    if accelerator is not None:
        accelerator.free_memory()

    model.to("cpu")
    logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated()}")
    logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved()}")

    return results


def run_test(args, **kwargs):
    accelerator = kwargs.get("accelerator")
    test_dataset = kwargs.get("test_dataset")
    test_dataset, le = prepare_dataset(args, test_dataset, include_targets=False)
    multi_task = not args.taskA

    # the final length of the input array, concatenating all modalities
    input_seq_len = get_input_length(args)
    if args.add_features:
        feature_space = le.classes_.size

    #  Model
    model = PerceiverPooler(
        text_model=args.text_model,
        vision_model=args.vision_model,
        n_tasks=5 if multi_task else 1,
        n_classes=2,
        input_size=768,
        input_seq_len=input_seq_len,
        add_feature_space=feature_space if args.add_features else None,
        add_caption=args.add_caption,
        depth=args.depth,
        num_latents=args.num_latents,
        latent_dim=args.latent_dim,
        cross_heads=1,
        latent_heads=8,
        cross_dim_head=64,
        latent_dim_head=64,
        weight_tie_layers=args.weight_tie,
        decoder_ff=True,
    )
    model.load_state_dict(torch.load(args.model_name_or_path))

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
            logits = model(**batch)
            logits = rearrange(logits, "b t c -> b c t")  # b x 2 x 5

        scores.append(logits.softmax(dim=1).detach().cpu())
        preds.append(logits.argmax(dim=1).detach().cpu())

    scores = torch.cat(scores).detach().cpu()
    preds = torch.cat(preds).detach().cpu()

    return {
        "scores": scores,
        "preds": preds,
    }
