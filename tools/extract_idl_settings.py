#!/usr/bin/env python3
"""Extract the per-night settings of the IDL predecessor's object blocks into JSON.

Usage
-----
    python3 tools/extract_idl_settings.py REDUCE_NOMIC_PRO OUT_JSON

reduce_nomic.pro sets each night's settings in an object block, ``if name eq '<night>' then
begin ... endif``. The script reads every block, splits its lines into statements (``&``
separates them; ``;`` starts a comment), and evaluates each assignment in order with a
restricted evaluator: numbers, strings, arrays, IDL index ranges (``[a:b]``), arithmetic,
comparisons, and references to settings set earlier in the block or to the defaults of
``reduce_nomic.config.DEFAULTS``. One-line ``if ... then ... [else ...]`` statements and
``if ... then begin ... endif`` groups are evaluated too, with ``testing``, ``debug`` and the
keyword arguments of the IDL procedure taken as 0. Names are lowercased, as IDL is
case-insensitive. A statement the evaluator cannot evaluate (a function call, an unknown
name) is kept verbatim under ``unevaluated`` for review.

Outputs
-------
OUT_JSON
    ``{night: {"lines": [first, last], "settings": {name: value}, "unevaluated": {name: text},
    "conditionals": [text]}}``, where ``lines`` are the block's 1-based line numbers and
    ``conditionals`` lists the ``if`` statements met, as written.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reduce_nomic.config import DEFAULTS  # noqa: E402

BLOCK = re.compile(r"^\s*if\s+name\s+eq\s+'([^']+)'\s+then\s+begin", re.I)
ASSIGN = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=(?!=)\s*(.+?)\s*$")
IF1 = re.compile(r"^\s*if\s+(.+?)\s+then\s+(.+?)(?:\s+else\s+(.+))?\s*$", re.I)
IFBEGIN = re.compile(r"^\s*if\s+(.+?)\s+then\s+begin\s*$", re.I)
OPS = [(r"\beq\b", "=="), (r"\bne\b", "!="), (r"\bge\b", ">="), (r"\ble\b", "<="), (r"\bgt\b", ">"),
       (r"\blt\b", "<"), (r"\band\b", " and "), (r"\bor\b", " or "), (r"\bnot\b", " not ")]


def strip_comment(line: str) -> str:
    """Remove an IDL comment, ignoring ``;`` inside quoted strings.

    Parameters
    ----------
    line : str
        One source line.

    Returns
    -------
    code : str
        The line without its comment.
    """
    out, quote = [], None
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
            out.append(ch)
        elif ch == ";":
            break
        else:
            out.append(ch)
    return "".join(out)


def split_statements(code: str) -> List[str]:
    """Split a line at the IDL statement separator ``&``, outside quoted strings.

    Parameters
    ----------
    code : str
        One line without its comment.

    Returns
    -------
    statements : list of str
        The non-empty statements.
    """
    parts, cur, quote = [], [], None
    for ch in code:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
            cur.append(ch)
        elif ch == "&":
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def evaluate(expr: str, env: Dict[str, Any]) -> Any:
    """Evaluate an IDL expression made of literals, arithmetic and known names.

    Parameters
    ----------
    expr : str
        The expression, as written in IDL.
    env : dict
        Known values by lowercased name.

    Returns
    -------
    value : any
        The value. Raises ``ValueError`` if the expression uses anything else.
    """
    text = expr.strip()
    strings: List[str] = []

    def stash(m):
        strings.append(m.group(0)[1:-1])
        return f"__S{len(strings) - 1}__"

    text = re.sub(r"'[^']*'|\"[^\"]*\"", stash, text)
    if re.search(r"\b[A-Za-z_][A-Za-z0-9_]*\s*\(", text):
        raise ValueError("function call")
    text = re.sub(r"!PI\b", "PI_", text, flags=re.I)
    text = re.sub(r"!DTOR\b", "DTOR_", text, flags=re.I)
    text = re.sub(r"\[\s*(-?\d+)\s*:\s*(-?\d+)\s*\]", lambda m: f"list(range({m.group(1)},{int(m.group(2)) + 1}))", text)
    for pat, rep in OPS:
        text = re.sub(pat, rep, text, flags=re.I)
    names = {k.lower(): v for k, v in env.items()}
    names.update(PI_=math.pi, DTOR_=math.pi / 180.0, list=list, range=range)
    keep = ("list", "range", "and", "or", "not", "PI_", "DTOR_")
    lowered = re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)\b",
                     lambda m: m.group(1) if m.group(1) in keep or re.fullmatch(r"__S\d+__", m.group(1)) else m.group(1).lower(),
                     text)
    lowered = re.sub(r"__S(\d+)__", lambda m: repr(strings[int(m.group(1))]), lowered)
    try:
        value = eval(lowered, {"__builtins__": {}}, names)  # noqa: S307 - restricted namespace
    except Exception as exc:  # noqa: BLE001
        raise ValueError(str(exc)) from exc
    if isinstance(value, tuple):
        value = list(value)
    return value


def run_statement(stmt: str, env: Dict[str, Any], out: Dict[str, Any]) -> None:
    """Evaluate one statement of an object block and record its effect.

    Parameters
    ----------
    stmt : str
        The statement.
    env : dict
        Values known so far, updated in place.
    out : dict
        The block's record (``settings``, ``unevaluated``, ``conditionals``), updated in place.

    Returns
    -------
    None
        ``env`` and ``out`` are updated in place.
    """
    m = IF1.match(stmt)
    if m and not IFBEGIN.match(stmt):
        out["conditionals"].append(stmt)
        try:
            cond = evaluate(m.group(1), env)
        except ValueError:
            out["unevaluated"][f"if:{stmt}"] = stmt
            return
        branch = m.group(2) if cond else m.group(3)
        if branch:
            run_statement(branch, env, out)
        return
    m = ASSIGN.match(stmt)
    if not m:
        return
    name, expr = m.group(1).lower(), m.group(2)
    try:
        value = evaluate(expr, env)
    except ValueError:
        out["unevaluated"][name] = expr
        return
    env[name] = value
    out["settings"][name] = value
    out["unevaluated"].pop(name, None)


def extract(path: str) -> Dict[str, Dict[str, Any]]:
    """Extract every object block of reduce_nomic.pro.

    Parameters
    ----------
    path : str
        Path of reduce_nomic.pro.

    Returns
    -------
    blocks : dict
        The records described in the module's Outputs, by night name.
    """
    src = open(path, errors="replace").read().splitlines()
    starts = [(i, BLOCK.match(l).group(1)) for i, l in enumerate(src) if BLOCK.match(l)]
    base = {k.lower(): v for k, v in DEFAULTS.items()}
    base.update(testing=0, debug=0, init=0, contrast=0, rho=0, theta=0, klip=0, adi=0, rdiklip=0)
    blocks: Dict[str, Dict[str, Any]] = {}
    for k, (i0, name) in enumerate(starts):
        depth, i = 0, i0 + 1
        body: List[Tuple[int, str]] = []
        while i < len(src):
            code = strip_comment(src[i]).strip()
            if re.match(r"^(if\b.*\bthen\s+begin|.*\bbegin)\s*$", code, re.I):
                depth += 1
            if re.match(r"^end(if|else)?\b", code, re.I):
                if depth == 0:
                    break
                depth -= 1
            body.append((i, code))
            i += 1
        env = dict(base)
        env["name"] = name
        rec = {"lines": [i0 + 1, i + 1], "settings": {}, "unevaluated": {}, "conditionals": []}
        skip_depth = None
        for _, code in body:
            if not code:
                continue
            mb = IFBEGIN.match(code)
            if mb:
                rec["conditionals"].append(code)
                try:
                    cond = evaluate(mb.group(1), env)
                except ValueError:
                    rec["unevaluated"][f"if:{code}"] = code
                    cond = False
                if not cond and skip_depth is None:
                    skip_depth = 1
                elif skip_depth is not None:
                    skip_depth += 1
                continue
            if re.match(r"^end(if|else)?\b", code, re.I):
                if skip_depth is not None:
                    skip_depth -= 1
                    if skip_depth == 0:
                        skip_depth = None
                continue
            if skip_depth is not None:
                continue
            for stmt in split_statements(code):
                run_statement(stmt, env, rec)
        blocks[name] = rec
    return blocks


def main(argv: List[str]) -> int:
    """Extract the blocks and write the JSON.

    Parameters
    ----------
    argv : list of str
        ``REDUCE_NOMIC_PRO OUT_JSON``.

    Returns
    -------
    status : int
        0 on success, 2 on a usage error.
    """
    if len(argv) != 2:
        print(__doc__)
        return 2
    blocks = extract(argv[0])
    with open(argv[1], "w") as fh:
        json.dump(blocks, fh, indent=1, sort_keys=True)
    n_un = sum(1 for b in blocks.values() for k in b["unevaluated"] if k != "root_dir")
    print(f"{len(blocks)} object blocks -> {argv[1]}; {n_un} unevaluated statements besides root_dir")
    for night, b in blocks.items():
        un = {k: v for k, v in b["unevaluated"].items() if k != "root_dir"}
        if un:
            print(f"  {night}: {un}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
