"""Guard against TUI Screen subclasses shadowing MessagePump internals.

In May 2026 we hit a multi-hour freeze bug: ``alignment_view.py`` and
``embedding_pca.py`` both initialised ``self._closing = False`` as a
custom flag for their worker threads. Textual's ``MessagePump`` base
class (which every Screen inherits from) ALSO has a ``self._closing``
attribute — it's the flag that ``post_message`` checks before queueing.

When ``action_back`` set ``self._closing = True``, it silently clobbered
MessagePump's flag, causing every subsequent ``post_message`` call —
including the ``Prune`` message that would have torn down the screen
— to return ``False`` without queueing. Alignment screen's pump
blocked forever at ``Queue.get`` waiting for a message that had been
silently dropped. ``screen.remove()`` awaited the pump → ``do_pop``
awaited that → the App's dispatch coroutine wedged → input died.

This test enumerates every ``textual.screen.Screen`` subclass we ship
and asserts it doesn't define a class- or instance-level attribute
whose name collides with the MessagePump reserved set. If anyone
adds a Screen with ``self._closing`` again, this test fails at PR
time instead of in production after a 4-hour debugging session.
"""

from __future__ import annotations

import inspect
import pkgutil
import importlib

import pytest


# Names we must NOT shadow on Screen subclasses. Sourced by inspecting
# textual's MessagePump and Widget bases — any private attribute they
# set in ``__init__`` is fair game for an accidental override.
RESERVED_NAMES = {
    "_closing",
    "_closed",
    "_running",
    "_task",
    "_message_queue",
    "_pending_message",
    "_thread_id",
    "_timers",
    "_next_callbacks",
    "_pruning",
    "_message_loop_task",
}


def _walk_tui_screen_classes():
    """Yield every ``Screen`` subclass declared under ``beak.tui.screens``."""
    from textual.screen import Screen
    import beak.tui.screens as screens_pkg

    for _finder, modname, _ispkg in pkgutil.walk_packages(
        screens_pkg.__path__, prefix="beak.tui.screens."
    ):
        try:
            mod = importlib.import_module(modname)
        except Exception:  # pragma: no cover - skip optional/broken modules
            continue
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if not issubclass(obj, Screen):
                continue
            if obj is Screen:
                continue
            # Only consider classes DEFINED in this module (not re-exports).
            if obj.__module__ != modname:
                continue
            yield obj


def _init_assigned_names(cls) -> set[str]:
    """Return every ``self.<name>`` assigned in ``cls.__init__``.

    Static parse of the AST — no need to instantiate the class (which
    is impractical for Textual screens since they need a running App).
    """
    import ast
    import textwrap

    init = cls.__dict__.get("__init__")
    if init is None:
        return set()
    try:
        src = textwrap.dedent(inspect.getsource(init))
    except (OSError, TypeError):
        return set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set()

    assigned: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Attribute) and \
                    isinstance(target.value, ast.Name) and \
                    target.value.id == "self":
                assigned.add(target.attr)
        # ``self.x: T = ...`` is an AnnAssign, not Assign.
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and \
                isinstance(node.target, ast.Attribute) and \
                isinstance(node.target.value, ast.Name) and \
                node.target.value.id == "self":
            assigned.add(node.target.attr)
    return assigned


@pytest.mark.parametrize("screen_cls", list(_walk_tui_screen_classes()))
def test_screen_does_not_shadow_message_pump_reserved_names(screen_cls):
    """Catch ``self._closing = ...`` and friends before they ship."""
    assigned = _init_assigned_names(screen_cls)
    collisions = assigned & RESERVED_NAMES
    assert not collisions, (
        f"{screen_cls.__module__}.{screen_cls.__name__} shadows "
        f"MessagePump-reserved attribute(s): {sorted(collisions)}. "
        f"This silently breaks Textual internals — see "
        f"tests/test_screen_attribute_collisions.py docstring for the "
        f"incident background. Rename to ``_user_<name>`` or similar."
    )
