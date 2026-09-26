"""Fast dataset-free smoke tests for the episodic heads."""

from __future__ import annotations

import sys
import types

import torch
import torch.nn as nn


class _FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_features = 384

    def forward(self, x):
        if x.ndim == 2 and x.shape[-1] == 384:
            return x
        return torch.zeros((x.shape[0], 384), device=x.device, dtype=x.dtype)


def _install_fake_timm() -> None:
    fake = types.ModuleType("timm")
    fake.create_model = lambda *args, **kwargs: _FakeEncoder()
    sys.modules["timm"] = fake


def main() -> None:
    _install_fake_timm()
    from models.wipt import WIPT
    from models.factorial import FactorialFewShotModel
    from train.domain_shift import shift_episode

    torch.manual_seed(123)
    support = torch.randn(25, 384)
    query = torch.randn(75, 384)

    q1 = WIPT(pretrained=False, query_group_size=1).eval()
    with torch.no_grad():
        scores, tq, ts, prototypes = q1.forward_from_embeddings(support, query, 5, 5)
        # Reconstruct the exact single-query computation.
        tokens = torch.cat(
            [query.unsqueeze(1), support.unsqueeze(0).expand(query.shape[0], -1, -1)],
            dim=1,
        )
        for layer in q1.transformer_layers:
            tokens = layer(tokens)
        tokens = q1.norm(tokens)
        manual_q = tokens[:, 0]
        manual_s = tokens[:, 1:].view(75, 5, 5, -1)
        manual_p = manual_s.mean(2)
        manual_scores = -torch.cdist(manual_q.unsqueeze(1), manual_p).squeeze(1)
    assert torch.equal(scores, manual_scores), "q=1 WIPT changed numerically"
    assert (
        torch.equal(tq, manual_q)
        and torch.equal(ts, manual_s)
        and torch.equal(prototypes, manual_p)
    )

    q5 = WIPT(pretrained=False, query_group_size=5).eval()
    q5.load_state_dict(q1.state_dict(), strict=True)
    with torch.no_grad():
        scores5, tq5, ts5, p5 = q5.forward_from_embeddings(support, query, 5, 5)
    assert (
        scores5.shape == (75, 5)
        and tq5.shape == (75, 384)
        and ts5.shape == (75, 5, 5, 384)
        and p5.shape == (75, 5, 384)
    )

    for adaptation in ["none", "support", "joint"]:
        for scorer in ["euclidean", "cosine", "relation"]:
            model = FactorialFewShotModel(
                adaptation=adaptation, scorer=scorer, pretrained=False
            ).eval()
            with torch.no_grad():
                out = model.forward_from_embeddings(support, query, 5, 5)
            assert out.shape == (75, 5) and torch.isfinite(out).all()

    support_images = torch.randn(25, 3, 32, 32)
    query_images = torch.randn(75, 3, 32, 32)
    shifted_s, shifted_q, mode = shift_episode(
        support_images, query_images, mode="mixed", strength=0.7
    )
    assert (
        shifted_s.shape == support_images.shape
        and shifted_q.shape == query_images.shape
    )
    assert (
        torch.isfinite(shifted_s).all()
        and torch.isfinite(shifted_q).all()
        and mode in {"shared", "cross"}
    )

    print("model checks passed")


if __name__ == "__main__":
    main()
