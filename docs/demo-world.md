# Synthetic demo world

`tests/fixtures/demo_world/v1.json` is public, deterministic test content. It contains one
organization, the stable `technology-ls` and `conservative` strategies, and intentionally
contradictory synthetic views of NVDA.

The loader treats IDs and the fixed clock as fixture inputs. Thesis, constraint, and asset-view
text is inert data: it cannot select a strategy, change a `ScopeKey`, grant access, or authorize an
effect. The fixture is not portfolio advice, production state, or a source for live holdings.
