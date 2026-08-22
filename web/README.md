# Future Leo web boundary

The web surface is intentionally a placeholder in this milestone. A future read-only UI may call
the versioned API under `src/leo/api`, but it must not access the privileged database credential,
reimplement scope resolution, or own task completion. No frontend dependencies are introduced by
the demo scaffold.
