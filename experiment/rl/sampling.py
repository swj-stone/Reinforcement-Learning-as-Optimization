"""Seeded balanced episode blocks and bounded source-specific replay pools."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from experiment.rl.environment import State

SOURCES = ("synthetic", "real")


class BalancedSampler:
    """Visit each available family/alpha cell once per source-size cycle."""

    def __init__(self, instances: list[dict], seed: int):
        self.rng = random.Random(seed)
        pools = defaultdict(lambda: defaultdict(list))
        seen = set()
        for instance in instances:
            source, n = instance.get("source"), instance.get("n")
            if source not in SOURCES or type(n) is not int or n <= 0:
                raise ValueError("sampler requires known sources and positive integer sizes")
            key = instance["instance_id"]
            if key in seen:
                raise ValueError(f"duplicate sampler instance ID: {key}")
            seen.add(key)
            cell = (str(instance.get("family") or "real"), float(instance["alpha"]))
            pools[(source, n)][cell].append(instance)
        sizes = {source: {n for s, n in pools if s == source} for source in SOURCES}
        if not sizes["synthetic"] or sizes["synthetic"] != sizes["real"]:
            raise ValueError("balanced blocks need matching available sizes for both sources")
        self.pools = {
            pair: {cell: sorted(records, key=lambda x: x["instance_id"])
                   for cell, records in sorted(cells.items())}
            for pair, cells in sorted(pools.items())
        }
        self.cycles = {pair: list(cells) for pair, cells in self.pools.items()}
        self.positions = dict.fromkeys(self.pools, 0)
        for cycle in self.cycles.values():
            self.rng.shuffle(cycle)

    def next_block(self) -> list[dict]:
        block = []
        for pair, cells in self.pools.items():
            position = self.positions[pair]
            cycle = self.cycles[pair]
            block.append(self.rng.choice(cells[cycle[position]]))
            position += 1
            if position == len(cycle):
                position = 0
                self.rng.shuffle(cycle)
            self.positions[pair] = position
        self.rng.shuffle(block)
        return block


@dataclass(frozen=True, slots=True)
class Transition:
    state: State
    action: int
    reward: float
    next_state: State | None


class BalancedReplay:
    """Two ring buffers; total retained transitions never exceeds capacity."""

    def __init__(self, capacity: int, seed: int):
        if type(capacity) is not int or capacity < 2 or capacity % 2:
            raise ValueError("replay capacity must be a positive even integer >= 2")
        self.capacity_per_source = capacity // 2
        self.rng = random.Random(seed)
        self.pools: dict[str, list[Transition]] = {source: [] for source in SOURCES}
        self.positions = dict.fromkeys(SOURCES, 0)

    @property
    def counts(self) -> dict[str, int]:
        return {source: len(pool) for source, pool in self.pools.items()}

    def add(self, transition: Transition) -> None:
        source = transition.state.instance["source"]
        if source not in self.pools:
            raise ValueError(f"unknown replay source: {source}")
        pool = self.pools[source]
        if len(pool) < self.capacity_per_source:
            pool.append(transition)
        else:
            pool[self.positions[source]] = transition
        self.positions[source] = (self.positions[source] + 1) % self.capacity_per_source

    def _validate_quotas(self, per_source: Mapping[str, int]) -> None:
        if (set(per_source) != set(SOURCES) or
                any(type(n) is not int or n < 0 for n in per_source.values())):
            raise ValueError("replay quotas require nonnegative integer counts for both sources")

    def ready(self, per_source: Mapping[str, int]) -> bool:
        self._validate_quotas(per_source)
        return all(len(self.pools[source]) >= count for source, count in per_source.items())

    def sample(self, per_source: Mapping[str, int]) -> list[Transition]:
        if not self.ready(per_source):
            raise ValueError("replay source pool has insufficient transitions")
        batch = [transition for source in SOURCES
                 for transition in self.rng.sample(self.pools[source], per_source[source])]
        self.rng.shuffle(batch)
        return batch
