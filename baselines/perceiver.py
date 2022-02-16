from turtle import forward
from pyrsistent import freeze
from transformers import AutoModel
import torch.nn as nn
import torch
from perceiver_pytorch import PerceiverIO
from einops import rearrange


class FinalProj(nn.Module):
    def __init__(self, n_tasks, queries_dim, logits_dim):
        super().__init__()

        self.projs = nn.ModuleList([])
        for i in range(n_tasks):
            self.projs.append(nn.Linear(queries_dim, logits_dim))

    def forward(self, latents):
        logits = list()
        n_tasks = latents.shape[1]
        for i in range(n_tasks):
            log = self.projs[i](latents[:, i, :])
            logits.append(log)

        logits = torch.stack(logits, axis=1)
        return logits


class PerceiverPooler(nn.Module):
    def __init__(
        self,
        text_model,
        vision_model,
        n_tasks,
        n_classes,
        input_size,
        input_seq_len,
        add_pos_emb=False,
        add_feature_space=None,
        add_caption=None,
        add_web_entities=None,
        use_separate_projs=False,
        **kwargs
    ):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(text_model, add_pooling_layer=False)
        self.vision_model = AutoModel.from_pretrained(
            vision_model, add_pooling_layer=False
        )

        self.input_seq_len = input_seq_len

        self.decoder_queries = nn.Parameter(torch.randn(n_tasks, input_size))

        if add_feature_space:
            self.feature_emb = nn.Embedding(add_feature_space, input_size)

        if add_caption:
            self.caption_encoder = AutoModel.from_pretrained("distilbert-base-uncased")

        if add_web_entities:
            self.entities_encoder = AutoModel.from_pretrained("distilbert-base-uncased")

        self.add_pos_emb = add_pos_emb
        if add_pos_emb:
            self.pos_emb = nn.Embedding(input_seq_len, input_size)

        self.use_separate_projs = use_separate_projs
        if use_separate_projs:
            self.pooler = PerceiverIO(dim=input_size, queries_dim=input_size, **kwargs)
            self.final_proj = FinalProj(n_tasks, input_size, n_classes)
        else:
            self.pooler = PerceiverIO(
                dim=input_size, queries_dim=input_size, logits_dim=n_classes, **kwargs
            )

        self.has_frozen_encoders = False

    def forward(
        self,
        pixel_values,
        input_ids,
        attention_mask,
        features=None,
        feature_mask=None,
        caption=None,
        caption_mask=None,
        entities=None,
        entities_mask=None,
        **kwargs
    ):
        b = pixel_values.shape[0]
        device = pixel_values.device

        v_out = self.vision_model(pixel_values=pixel_values)
        t_out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)

        v_emb = v_out.last_hidden_state[:, 1:, :]  # drop [CSL] introduced by ViT
        t_emb = t_out.last_hidden_state

        inputs = [v_emb, t_emb]

        # concat feature embeddings
        if (
            hasattr(self, "feature_emb")
            and features is not None
            and feature_mask is not None
        ):
            f_emb = self.feature_emb(features)
            inputs.append(f_emb)

        # concat caption embeddings
        if (
            hasattr(self, "caption_encoder")
            and caption is not None
            and caption_mask is not None
        ):
            c_emb = self.caption_encoder(input_ids=caption, attention_mask=caption_mask)
            c_emb = c_emb.last_hidden_state
            inputs.append(c_emb)

        # concat web entities embeddings
        if (
            hasattr(self, "entities_encoder")
            and entities is not None
            and entities_mask is not None
        ):
            e_emb = self.entities_encoder(
                input_ids=entities, attention_mask=entities_mask
            )
            e_emb = e_emb.last_hidden_state
            inputs.append(e_emb)

        in_array = torch.cat(inputs, axis=1)

        if self.add_pos_emb:
            pos_emb = self.pos_emb(torch.arange(in_array.shape[1], device=device))
            pos_emb = rearrange(pos_emb, "seq d -> () seq d")
            in_array = in_array + pos_emb

        v_mask = torch.ones(b, v_emb.shape[1], device=device)  # b seq h
        masks = [v_mask, attention_mask]

        if features is not None:
            masks.append(feature_mask)

        if caption is not None:
            masks.append(caption_mask)

        if entities is not None:
            masks.append(entities_mask)

        mask = torch.cat(masks, axis=-1).bool()

        logits = self.pooler(in_array, mask=mask, queries=self.decoder_queries)

        if self.use_separate_projs:
            logits = self.final_proj(logits)

        return logits

    def _set_encoders(self, freeze):
        for p in self.vision_model.parameters():
            p.requires_grad = not freeze

        for p in self.text_model.parameters():
            p.requires_grad = not freeze

        if hasattr(self, "caption_encoder"):
            for p in self.caption_encoder.parameters():
                p.requires_grad = not freeze

    def freeze_encoders(self):
        self._set_encoders(freeze=True)
        self.has_frozen_encoders = True

    def unfreeze_encoders(self):
        self._set_encoders(freeze=False)
        self.has_frozen_encoders = False
