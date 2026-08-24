"""Leo's agent: a ReAct loop, its tools, its memory, and its durable trace.

The design rule for this package is that nothing in it may overrule the model.
The loop supplies context, runs the tools the model asks for, keeps both within
budget, and stores what happened. The model decides what to do and writes the
answer.

Import submodules directly (``from leo.agent.runtime import runtime``). This
module stays free of imports so that ``leo.agent.contracts`` -- which the
provider adapters depend on -- never pulls the runtime in behind it.
"""
