"""Console output that cannot crash a run on a narrow Windows code page.

When stdout is redirected to a file or a pipe - which is exactly what
run_visual_experiment.py does for every train.log / eval.log - Windows Python
encodes it with the ANSI code page (cp1252 on most Western machines), and any
character outside that code page raises UnicodeEncodeError from inside print().
eval_gt_relations.py died that way on a U+2229 in its overlap-check line, after
the checkpoint, the relationships and the frozen split had all loaded.

make_stream_safe() keeps the stream's encoding and changes only what happens to
a character that encoding cannot represent: it is written as a backslash escape
(e.g. \\u2229) instead of raising. This is the policy Python already applies to
stderr. It does not catch, hide or alter any other exception.
"""
from __future__ import annotations

import sys


def make_stream_safe(stream) -> bool:
    """Switch a text stream to errors="backslashreplace". Returns True if applied.

    Streams that are not io.TextIOWrapper instances (pytest capture objects,
    some IDE consoles) have no reconfigure() and are left untouched.
    """
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return False
    reconfigure(errors="backslashreplace")
    return True


def configure_safe_stdio() -> None:
    """Apply make_stream_safe to sys.stdout and sys.stderr. Call once at startup."""
    for stream in (sys.stdout, sys.stderr):
        if stream is not None:
            make_stream_safe(stream)
