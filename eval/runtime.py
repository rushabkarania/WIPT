"""Shared model loading and episodic evaluation helpers."""

from __future__ import annotations

from pathlib import Path

import torch

from configs.experiment import (
    BACKBONE,
    CHECKPOINTS,
    N_QUERY,
    NUM_WORKERS,
    N_SHOT,
    N_WAY,
    WIPT_DROPOUT,
    WIPT_HEADS,
)
from models.baselines import CosineMatchingViT, SupportTransformerViT, RelationHeadViT
from models.protonet import ProtoNet
from models.wipt import WIPT
from models.wipt_variants import RefinedWIPT
from utils.checkpoints import (
    encoder_state_from_checkpoint,
    load_checkpoint,
    load_wipt_state_compat,
    model_state,
    verify_encoder,
)
from utils.data import evaluation_transform, make_episode_loader
from utils.metrics import set_seed


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def episode_loader(
    data_path,
    seed: int,
    n_episodes: int,
    *,
    transform=None,
    n_way: int = N_WAY,
    n_shot: int = N_SHOT,
    n_query: int = N_QUERY,
    shuffle_queries: bool = False,
):
    set_seed(seed)
    return make_episode_loader(
        data_path,
        n_way=n_way,
        n_shot=n_shot,
        n_query=n_query,
        n_episodes=n_episodes,
        transform=transform or evaluation_transform(),
        num_workers=NUM_WORKERS,
        seed=seed,
        shuffle_queries=shuffle_queries,
    )


def load_reference_wipt(device, checkpoint_path: str | Path | None = None):
    path = Path(checkpoint_path or CHECKPOINTS["WIPT-2"])
    checkpoint = load_checkpoint(path, device="cpu")
    layers = int(checkpoint.get("num_layers", 2))
    model = WIPT(
        backbone=BACKBONE,
        pretrained=False,
        num_layers=layers,
        num_heads=WIPT_HEADS,
        dropout=WIPT_DROPOUT,
    )
    load_wipt_state_compat(model, checkpoint)
    return model.to(device).eval(), checkpoint


def reference_encoder_state(checkpoint_path: str | Path | None = None):
    checkpoint = load_checkpoint(checkpoint_path or CHECKPOINTS["WIPT-2"], device="cpu")
    return encoder_state_from_checkpoint(checkpoint)


def load_frozen_protonet(device, reference_checkpoint: str | Path | None = None):
    encoder_state = reference_encoder_state(reference_checkpoint)
    model = (
        ProtoNet(backbone=BACKBONE, encoder_state_dict=encoder_state).to(device).eval()
    )
    verify_encoder(model, encoder_state)
    return model


def _load_state(model, checkpoint, *, strip_unused_cls_token: bool = False):
    if strip_unused_cls_token:
        load_wipt_state_compat(model, checkpoint)
        return
    state = dict(model_state(checkpoint))
    state.pop("cls_token", None)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )


def load_paper_models(device, *, include_variants: bool = True):
    """Load the paper models and verify their frozen encoders against WIPT-2."""
    reference_state = reference_encoder_state()
    models = {}

    protonet = (
        ProtoNet(backbone=BACKBONE, encoder_state_dict=reference_state)
        .to(device)
        .eval()
    )
    verify_encoder(protonet, reference_state)
    models["ProtoNet+ViT"] = (protonet, "protonet")

    matching = (
        CosineMatchingViT(
            backbone=BACKBONE,
            pretrained=False,
            encoder_state_dict=reference_state,
        )
        .to(device)
        .eval()
    )
    verify_encoder(matching, reference_state)
    models["CosineMatching+ViT"] = (matching, "cosine_matching")

    for name, constructor, kind in [
        ("RelationHead+ViT", RelationHeadViT, "relationhead"),
        ("SupportTransformer+ViT", SupportTransformerViT, "support_transformer"),
    ]:
        path = Path(CHECKPOINTS[name])
        if not path.exists():
            raise FileNotFoundError(f"Required paper checkpoint is missing: {path}")
        checkpoint = load_checkpoint(path, device="cpu")
        kwargs = {"backbone": BACKBONE, "pretrained": False}
        if name == "SupportTransformer+ViT":
            kwargs["num_layers"] = 2
        model = constructor(**kwargs)
        _load_state(model, checkpoint)
        verify_encoder(model, reference_state)
        models[name] = (model.to(device).eval(), kind)

    reference_model, _ = load_reference_wipt(device)
    verify_encoder(reference_model, reference_state)
    models["WIPT-2"] = (reference_model, "wipt")

    if include_variants:
        for name, layers in [("WIPT-4", 4), ("WIPT-6", 6)]:
            path = Path(CHECKPOINTS[name])
            if not path.exists():
                raise FileNotFoundError(f"Required paper checkpoint is missing: {path}")
            checkpoint = load_checkpoint(path, device="cpu")
            model = WIPT(
                backbone=BACKBONE,
                pretrained=False,
                num_layers=int(checkpoint.get("num_layers", layers)),
                num_heads=WIPT_HEADS,
                dropout=WIPT_DROPOUT,
            )
            _load_state(model, checkpoint, strip_unused_cls_token=True)
            verify_encoder(model, reference_state)
            models[name] = (model.to(device).eval(), "wipt")

        for name in ["WIPT-Residual", "WIPT-All"]:
            path = Path(CHECKPOINTS[name])
            if not path.exists():
                raise FileNotFoundError(f"Required paper checkpoint is missing: {path}")
            checkpoint = load_checkpoint(path, device="cpu")
            kwargs = dict(checkpoint.get("refinement_kwargs", {}))
            if "attn_temperature" in kwargs:
                kwargs["attention_temperature"] = kwargs.pop("attn_temperature")
            model = RefinedWIPT(
                backbone=BACKBONE,
                pretrained=False,
                num_layers=int(checkpoint.get("num_layers", 2)),
                **kwargs,
            )
            _load_state(model, checkpoint)
            verify_encoder(model, reference_state)
            models[name] = (model.to(device).eval(), "wipt_refined")

    return models
