"""Strict loader for the immutable synthetic two-strategy demo world."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DemoAssetView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    asset_id: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    stance: str = Field(min_length=1)
    thesis: str = Field(min_length=1)
    target_weight: float = Field(ge=0, le=1)


class DemoStrategy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    thesis: str = Field(min_length=1)
    constraint: str = Field(min_length=1)
    asset_views: tuple[DemoAssetView, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def asset_ids_are_unique(self) -> DemoStrategy:
        asset_ids = [view.asset_id for view in self.asset_views]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("strategy asset IDs must be unique")
        return self


class DemoWorld(BaseModel):
    """Versioned synthetic content; scope IDs are fixture authority, text is inert data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["v1"]
    fixed_now: datetime
    organization_id: str = Field(min_length=1)
    strategies: tuple[DemoStrategy, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_world(self) -> DemoWorld:
        strategy_ids = [strategy.id for strategy in self.strategies]
        if len(strategy_ids) != len(set(strategy_ids)):
            raise ValueError("strategy IDs must be unique")
        if len(self.strategies) != 2:
            raise ValueError("demo world must contain exactly two strategies")
        for strategy in self.strategies:
            for view in strategy.asset_views:
                if not view.symbol.strip():
                    raise ValueError(f"asset view {view.asset_id!r} has an empty symbol")
        return self

    def strategy(self, strategy_id: str) -> DemoStrategy:
        for strategy in self.strategies:
            if strategy.id == strategy_id:
                return strategy
        raise KeyError(strategy_id)


def load_demo_world(source: Mapping[str, Any] | str | Path) -> DemoWorld:
    """Load and validate a fixture from a mapping, JSON string, or JSON path."""

    if isinstance(source, Path):
        payload = json.loads(source.read_text(encoding="utf-8"))
    elif isinstance(source, str):
        payload = json.loads(source)
    else:
        payload = dict(source)
    if not isinstance(payload, dict):
        raise ValueError("demo world JSON root must be an object")
    return DemoWorld.model_validate(payload)
