"""Provider-neutral errors for Leo's atomic run-store contract."""


class StoreError(RuntimeError):
    """Base persistence error safe for harness control flow."""


class NotFoundError(StoreError):
    pass


class ConcurrencyError(StoreError):
    pass
