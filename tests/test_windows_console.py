"""Evaluation and training logs must survive a Windows cp1252 console.

Regression cover for: eval_gt_relations.py printed U+2229 (the intersection
sign) in its overlap-check line. run_visual_experiment.py redirects every
child's stdout to a log file, Windows encodes a redirected stream with the ANSI
code page (cp1252), and cp1252 has no U+2229 - so every evaluation died with
UnicodeEncodeError right after the frozen split loaded.
"""
from __future__ import annotations

import ast
import io
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.console import configure_safe_stdio, make_stream_safe  # noqa: E402

# Everything the GPU experiment executes, directly or by import, that prints.
RUN_PATH_SCRIPTS = [
    "eval_gt_relations.py",
    "train_full_visual_semantic.py",
    "run_visual_experiment.py",
    "build_clip_cache.py",
    "prepare_visual_genome.py",
    "check_environment.py",
    "relation_prediction/vg_dataset.py",
    "relation_prediction/clip_cache.py",
    "relation_prediction/clip_extractor.py",
    "relation_prediction/model.py",
]


def _print_literals(path):
    """(lineno, text) of every string literal passed to print() in a file."""
    with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "print"):
            continue
        for arg in node.args:
            for sub in ast.walk(arg):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    out.append((sub.lineno, sub.value))
    return out


def cp1252_problems(path):
    bad = []
    for lineno, text in _print_literals(path):
        try:
            text.encode("cp1252")
        except UnicodeEncodeError as exc:
            bad.append((lineno, text[exc.start:exc.end]))
    return bad


# --------------------------------------------------------------------------
# the line that crashed
# --------------------------------------------------------------------------

def test_overlap_line_is_ascii():
    from eval_gt_relations import assert_disjoint, format_overlap_line

    line = format_overlap_line(assert_disjoint([1, 2], [3], [4]))
    line.encode("ascii")                                   # raised before the fix
    assert "PASS" in line
    assert "train&val=0" in line and "train&test=0" in line and "val&test=0" in line


def test_overlap_line_prints_through_a_real_cp1252_stdout():
    """Reproduce the GPU machine: a child Python whose stdout is strict cp1252.

    Deliberately does NOT call configure_safe_stdio, so this pins the ASCII
    fix on its own.
    """
    code = ("import eval_gt_relations as e; "
            "print(e.format_overlap_line(e.assert_disjoint([1], [2], [3])))")
    env = {**os.environ, "PYTHONIOENCODING": "cp1252:strict", "PYTHONUTF8": "0"}
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                          capture_output=True, timeout=300)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    assert b"overlap check: PASS" in proc.stdout


# --------------------------------------------------------------------------
# no unencodable text anywhere on the run path
# --------------------------------------------------------------------------

def test_the_scan_detects_the_original_bug(tmp_path, monkeypatch):
    """Guard the guard: the scanner must flag the exact pre-fix line."""
    src = tmp_path / "old_eval.py"
    src.write_text('print(f"(train∩val={1})")\n', encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "ROOT", str(tmp_path))
    assert cp1252_problems("old_eval.py") == [(1, "∩")]


@pytest.mark.parametrize("path", RUN_PATH_SCRIPTS)
def test_run_path_print_literals_are_cp1252_encodable(path):
    assert cp1252_problems(path) == []


def test_evaluator_print_literals_are_pure_ascii():
    """The evaluator is the script that crashed; hold it to the stricter bar."""
    bad = [(ln, t) for ln, t in _print_literals("eval_gt_relations.py")
           if not t.isascii()]
    assert bad == []


# --------------------------------------------------------------------------
# the safety net for text we do not control (paths, labels, exception text)
# --------------------------------------------------------------------------

def _strict_cp1252_stream():
    raw = io.BytesIO()
    # newline="\n": compare bytes without Windows' \r\n translation.
    return raw, io.TextIOWrapper(raw, encoding="cp1252", errors="strict", newline="\n")


def test_strict_cp1252_stream_really_raises_without_the_fix():
    _, stream = _strict_cp1252_stream()
    with pytest.raises(UnicodeEncodeError):
        stream.write("train∩val")
        stream.flush()


def test_make_stream_safe_escapes_instead_of_raising():
    raw, stream = _strict_cp1252_stream()
    assert make_stream_safe(stream) is True
    stream.write("train∩val — ok\n")
    stream.flush()
    assert stream.encoding == "cp1252"                     # encoding untouched
    assert raw.getvalue() == b"train\\u2229val \x97 ok\n"   # cp1252 em dash kept


def test_make_stream_safe_leaves_non_textio_streams_alone():
    class NotTextIO:
        def write(self, s):
            return len(s)
    assert make_stream_safe(NotTextIO()) is False


def test_configure_safe_stdio_under_a_real_cp1252_child():
    code = ("from utils.console import configure_safe_stdio; "
            "configure_safe_stdio(); print('train\\u2229val')")
    env = {**os.environ, "PYTHONIOENCODING": "cp1252:strict", "PYTHONUTF8": "0"}
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                          capture_output=True, timeout=120)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    assert proc.stdout.strip() == b"train\\u2229val"


@pytest.mark.parametrize("script", ["eval_gt_relations.py",
                                    "train_full_visual_semantic.py",
                                    "run_visual_experiment.py"])
def test_entry_points_configure_safe_stdio(script):
    with open(os.path.join(ROOT, script), encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    main_guard = [n for n in tree.body if isinstance(n, ast.If)
                  and "__main__" in ast.unparse(n.test)]
    assert main_guard, f"{script} has no __main__ block"
    assert "configure_safe_stdio()" in ast.unparse(main_guard[0])


def test_runner_children_write_utf8_logs():
    from run_visual_experiment import child_env
    env = child_env()
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert env.get("PATH") == os.environ.get("PATH")       # inherits, not replaces
