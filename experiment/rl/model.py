"""Shared-item DQN, feasible epsilon-greedy actions and cached batch inference."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import torch
from torch import nn

from experiment.baselines.methods import BaselineSolution
from experiment.rl.environment import State, batch_states


class QNetwork(nn.Module):
    """The fixed 64/64 item encoder and 128-unit two-action head from the plan."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(133, 128), nn.ReLU(), nn.Linear(128, 2))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        embeddings = self.encoder(batch["items"])
        valid = batch["valid"].unsqueeze(-1)
        count = valid.sum(dim=1)
        if (count == 0).any():
            raise ValueError("Terminal observations cannot be pooled")
        mean = embeddings.masked_fill(~valid, 0).sum(dim=1) / count
        maximum = embeddings.masked_fill(~valid, -torch.inf).amax(dim=1)
        return self.head(torch.cat((mean, maximum, batch["context"]), dim=1))

    def suffix_pools(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode once and pool every suffix for use while the weights are fixed.

        Trailing padded positions have zero mean and max. Real positions match
        ``forward`` pooling, up to floating point summation roundoff. The caller
        owns the result; no cache survives a network update or a solution call.
        """
        embeddings = self.encoder(batch["items"])
        valid = batch["valid"].unsqueeze(-1)
        counts = valid.flip((1,)).cumsum(dim=1).flip((1,))
        sums = embeddings.masked_fill(~valid, 0).flip((1,)).cumsum(dim=1).flip((1,))
        maxima = (
            embeddings.masked_fill(~valid, -torch.inf).flip((1,))
            .cummax(dim=1).values.flip((1,))
        )
        return sums / counts.clamp_min(1), maxima.masked_fill(counts == 0, 0)


def _greedy_actions(q_values: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(q_values).all():
        raise ValueError("Policy Q values must be finite")
    # argmax returns the first maximum, so exact ties select skip (column zero).
    return q_values.masked_fill(~action_mask, -torch.inf).argmax(dim=1)


def select_action(
    network: QNetwork,
    state: State,
    epsilon: float,
    rng,
    device: str | torch.device = "cpu",
) -> int:
    """Choose only feasible actions for both exploration and exploitation."""
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0, 1]")
    batch = batch_states([state], device=device)
    if epsilon > 0 and rng.random() < epsilon:
        actions = batch["action_mask"][0].nonzero(as_tuple=False).flatten().tolist()
        return int(rng.choice(actions))
    with torch.no_grad():
        return int(_greedy_actions(network(batch), batch["action_mask"])[0].item())


@torch.no_grad()
def _solve_chunk(
    network: QNetwork, instances: Sequence[dict], device: str | torch.device
) -> list[BaselineSolution]:
    """Solve a same-size chunk with one encoding and one head call per position."""
    n = len(instances[0]["weights"])
    batch = batch_states([State(instance, 0, instance["capacity"]) for instance in instances], device)
    means, maxima = network.suffix_pools(batch)
    weights = torch.tensor(
        [[instance["weights"][i] for i in instance["item_order"]] for instance in instances],
        dtype=torch.int64, device=device,
    )
    remaining = torch.tensor(
        [instance["capacity"] for instance in instances], dtype=torch.int64, device=device,
    )
    denominators = remaining.clamp_min(1)
    selected = torch.zeros((len(instances), n), dtype=torch.bool, device=device)
    context = batch["context"].clone()
    mask = batch["action_mask"].clone()
    for position in range(n):
        context[:, :2] = batch["items"][:, position]
        context[:, 2] = remaining / denominators
        context[:, 3] = (n - position) / n
        mask[:, 1] = weights[:, position] <= remaining
        q_values = network.head(torch.cat((means[:, position], maxima[:, position], context), dim=1))
        actions = _greedy_actions(q_values, mask)
        selected[:, position] = actions.bool()
        remaining -= weights[:, position] * actions

    solutions = []
    for instance, selection in zip(instances, selected.cpu().tolist()):
        indices = tuple(sorted(
            index for index, include in zip(instance["item_order"], selection) if include
        ))
        solutions.append(BaselineSolution(
            indices,
            sum(instance["values"][index] for index in indices),
            sum(instance["weights"][index] for index in indices),
        ))
    return solutions


def solve_instances(
    network: QNetwork,
    instances: Sequence[dict],
    device: str | torch.device = "cpu",
) -> list[BaselineSolution]:
    """Return baseline-compatible solutions in input order, without exploration.

    Grouping by n avoids padding work; bounded chunks cap temporary embedding
    memory even for full validation corpora. Encoding is included in every call.
    """
    if not instances:
        return []
    grouped: dict[int, list[tuple[int, dict]]] = defaultdict(list)
    results = [BaselineSolution((), 0, 0)] * len(instances)
    for index, instance in enumerate(instances):
        n = len(instance["weights"])
        if n:
            grouped[n].append((index, instance))
    for group in grouped.values():
        for start in range(0, len(group), 128):
            chunk = group[start:start + 128]
            solutions = _solve_chunk(network, [instance for _, instance in chunk], device)
            for (index, _), solution in zip(chunk, solutions):
                results[index] = solution
    return results
