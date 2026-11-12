"""Optional token-aligned residual head; no data, models, or training are loaded."""
from __future__ import annotations

import torch
from torch import nn


class GenerationResidualFusion(nn.Module):
    """Add a gated correction to a separately available semantic risk logit.

    Inputs share leading dimensions (e.g. token or batch,token): hidden[...,768],
    baseline_logit[...], lookback[...,1024], nll[...,1]. Every position denotes
    the SAME original Llama token. Callers own alignment and eligible masks.
    """

    SEMANTIC_DIM = 768
    LOOKBACK_DIM = 1024
    FUSION_DIM = 65

    def __init__(self) -> None:
        super().__init__()
        self.semantic = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 32), nn.SiLU())
        self.lookback = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 24), nn.SiLU())
        self.nll = nn.Sequential(nn.Linear(1, 8), nn.SiLU())
        self.gate = nn.Linear(self.FUSION_DIM, 1)
        self.residual = nn.Linear(self.FUSION_DIM, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def components(
        self,
        semantic_hidden: torch.Tensor,
        baseline_logit: torch.Tensor,
        lookback: torch.Tensor,
        nll: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        leading = baseline_logit.shape
        if semantic_hidden.shape != (*leading, self.SEMANTIC_DIM):
            raise ValueError('semantic_hidden must match baseline positions and end in 768')
        if lookback.shape != (*leading, self.LOOKBACK_DIM):
            raise ValueError('lookback must match baseline positions and end in 1024')
        if nll.shape != (*leading, 1):
            raise ValueError('nll must match baseline positions and end in 1')
        if torch.any(nll < 0):
            raise ValueError('NLL must be nonnegative; no clamping or fitted scaling is applied')

        semantic = self.semantic(semantic_hidden)
        generation = torch.cat((self.lookback(lookback), self.nll(torch.log1p(nll))), dim=-1)
        fused = torch.cat((semantic, generation, baseline_logit.unsqueeze(-1)), dim=-1)
        gate = torch.sigmoid(self.gate(fused)).squeeze(-1)
        raw_residual = self.residual(fused).squeeze(-1)
        correction = gate * raw_residual
        return {'semantic': semantic, 'generation': generation, 'gate': gate,
                'raw_residual': raw_residual, 'correction': correction,
                'logit': baseline_logit + correction}

    def forward(
        self,
        semantic_hidden: torch.Tensor,
        baseline_logit: torch.Tensor,
        lookback: torch.Tensor,
        nll: torch.Tensor,
    ) -> torch.Tensor:
        return self.components(semantic_hidden, baseline_logit, lookback, nll)['logit']
