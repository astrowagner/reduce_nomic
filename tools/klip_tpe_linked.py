"""Run a klip-tpe command line with more of the partitions' parameters linked, one value for every partition.

usage: python tools/klip_tpe_linked.py --link BASE [BASE ...] -- {generic,near,resume,extend} ARGS ...

KLIP-TPE gives each partition (each chop of a night) its own block of reduction parameters and links
some of them, one value for every partition, through ``RunConfig.link_params`` (default ``bin`` and
``filter``): every proposal of the optimizer gives each partition the first partition's value of a
linked base. klip-tpe ``aa0702b`` has no command-line option for that list, so this wrapper runs the
klip-tpe command line after ``--`` with two changes:

* a new run takes ``--link`` as its ``link_params``, recorded in ``run_setup.txt`` and the
  checkpoint; a resumed run must hold the same list in its checkpoint, or the wrapper stops;
* the run's feasibility projection gets a last step, :class:`LinkTie`, which links the bases again
  after the guard. The reference-count guard projects ``angsep`` and ``anglemax`` partition by
  partition, so that every target frame keeps enough references, and partitions whose frames differ
  come out of it with different values (adi3ann3's warm-up evaluation 2 had anglemax 60 and 54).
  ``angsep`` takes the smallest of the selected partitions' values and ``anglemax`` the largest:
  references are only added as ``angsep`` falls or ``anglemax`` grows, so every partition keeps the
  references its own projection gave it. Every other base takes the first partition's value, as
  KLIP-TPE's own link does.

A run started through this wrapper is resumed through it too (``-- resume --instrument generic
...``): a plain ``klip-tpe resume`` keeps the checkpoint's ``link_params`` but not the last
projection step.

Parameters
----------
--link : str list
    The bases linked, such as ``bin filter angsep anglemax``; each must be a reduction parameter of
    the run's search space, or the wrapper stops before the search starts.
ARGS : str list
    The klip-tpe command line after ``--``, from its subcommand on, handed to ``klip_tpe.cli.main``.

Returns
-------
status : int
    0 once the klip-tpe command returns; klip-tpe's own errors stop the program as they do
    ``klip-tpe``.
run directory : files
    What the klip-tpe command writes, with ``link_params`` and the projection's ``link_tie`` step
    in ``run_setup.txt`` and ``run_setup.json``.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

SUBCOMMANDS = ("generic", "near", "resume", "extend")
_installed: Optional[List[str]] = None


class LinkTie:
    """Projection step that gives every partition the same value of each linked base.

    Parameters
    ----------
    bases : sequence of str
        The bases linked, in order; a base without partition replicas is left alone.

    Notes
    -----
    ``angsep`` takes the smallest value over the partitions the vector selects and ``anglemax`` the
    largest, the most permissive pair, so every selected partition keeps at least the references its
    own projection left it. Every other base takes the first partition's value
    (``SearchSpace.tie``). Partitions the vector drops get the same value, which they never use.
    """

    def __init__(self, bases: Sequence[str]):
        self.bases = [str(b) for b in dict.fromkeys(bases) if b]

    def __call__(self, x: np.ndarray, space: Any, **_: Any) -> np.ndarray:
        """Link the bases of one vector.

        Parameters
        ----------
        x : ndarray, shape (ndim,)
            The vector, after the run's guard.
        space : klip_tpe.space.SearchSpace
            The run's search space, whose ``ndim`` is the vector's length.
        **_ : dict
            The projection's other keywords (``rng``, ``is_random``, ``zone``), not used.

        Returns
        -------
        x : ndarray, shape (ndim,)
            A copy with each linked base the same in every partition.
        """
        x = np.asarray(x, dtype=float).copy()
        selected = set(space.selected_partitions(x)) if space.partitions else set()
        for b in self.bases:
            idx = [int(i) for i in space.dims_of_base(b) if space.params[int(i)].partition is not None]
            if len(idx) < 2:
                continue
            use = [i for i in idx if space.params[i].partition in selected] or idx
            if b == "angsep":
                v = float(np.min(x[use]))
            elif b == "anglemax":
                v = float(np.max(x[use]))
            else:
                v = float(x[idx[0]])
            x[idx] = v
        return x

    def describe(self) -> Dict[str, Any]:
        """Return the step's record for ``run_setup.json``.

        Returns
        -------
        record : dict
            ``name``, ``bases`` and the ``rule`` applied.
        """
        return {"name": "link_tie", "bases": list(self.bases),
                "rule": "angsep: smallest over the selected partitions; anglemax: largest; "
                        "other bases: the first partition's value"}


def install(bases: Sequence[str]) -> List[str]:
    """Make klip-tpe link ``bases`` in every run this process starts or resumes.

    It wraps ``klip_tpe.cli._near_objects``, so that the run's guard ends with :class:`LinkTie`, and
    ``klip_tpe.runner.Runner.__init__``, so that a new run's ``link_params`` is ``bases`` and a
    resumed run's must equal them.

    Parameters
    ----------
    bases : sequence of str
        The bases linked.

    Returns
    -------
    link : list of str
        The bases, each once, in order.

    Raises
    ------
    RuntimeError
        If it was installed already with other bases.
    SystemExit
        When a run starts or resumes: a base is not a reduction parameter of its search space, or a
        resumed run's checkpoint links other bases.
    """
    global _installed
    link = [str(b) for b in dict.fromkeys(bases) if b]
    if _installed is not None:
        if _installed != link:
            raise RuntimeError(f"already installed with {_installed}, not {link}")
        return link
    import klip_tpe.cli as cli
    from klip_tpe import runner as kr
    from klip_tpe.feasibility import compose

    near_objects = cli._near_objects

    def linked_near_objects(a, *args, **kw):
        """Build klip-tpe's objects, the guard ending with :class:`LinkTie`."""
        red, objective, sampler, guard, thr = near_objects(a, *args, **kw)
        return red, objective, sampler, compose(guard, LinkTie(link)), thr

    runner_init = kr.Runner.__init__

    def linked_runner_init(self, *args, **kw):
        """Start a runner with ``link_params`` set to the linked bases, or check a resumed run's."""
        space = args[1] if len(args) > 1 else kw.get("space")
        config = args[4] if len(args) > 4 else kw.get("config")
        missing = [b for b in link if b not in space.bases]
        if missing:
            raise SystemExit(f"--link {' '.join(missing)}: not a reduction parameter of this search "
                             f"(it has {', '.join(space.bases)})")
        if kw.get("_from_checkpoint"):
            have = [str(b) for b in (config.link_params or [])]
            if sorted(have) != sorted(link):
                raise SystemExit(f"the checkpoint links {have}, not --link {' '.join(link)}: resume with "
                                 f"--link {' '.join(have)}")
        else:
            config.link_params = list(link)
        runner_init(self, *args, **kw)

    cli._near_objects = linked_near_objects
    kr.Runner.__init__ = linked_runner_init
    _installed = link
    return link


def split_argv(argv: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Split a command line at the first ``--``.

    Parameters
    ----------
    argv : sequence of str
        The command line without the program's name.

    Returns
    -------
    own : list of str
        The wrapper's arguments, before ``--``.
    klip_tpe_args : list of str
        The klip-tpe command line, after ``--``.

    Raises
    ------
    SystemExit
        If there is no ``--``, or nothing after it.
    """
    argv = list(argv)
    if "--" not in argv:
        raise SystemExit("usage: klip_tpe_linked.py --link BASE [BASE ...] -- {generic,near,resume,extend} ARGS ...")
    i = argv.index("--")
    own, rest = argv[:i], argv[i + 1:]
    if not rest:
        raise SystemExit("give the klip-tpe command line after --")
    return own, rest


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run a klip-tpe command line with the linked bases (see the module docstring).

    Parameters
    ----------
    argv : sequence of str or None, default None
        The command line without the program's name. None takes ``sys.argv[1:]``.

    Returns
    -------
    status : int
        0 once the klip-tpe command returns.
    """
    own, rest = split_argv(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="klip_tpe_linked.py", description=__doc__.split("\n")[0])
    ap.add_argument("--link", nargs="+", required=True)
    a = ap.parse_args(own)
    bad = [b for b in a.link if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", b)]
    if bad:
        ap.error(f"--link takes parameter names, not {' '.join(bad)}")
    if rest[0] not in SUBCOMMANDS:
        ap.error(f"the klip-tpe command after -- starts with one of {', '.join(SUBCOMMANDS)}, not {rest[0]}")
    link = install(a.link)
    import klip_tpe
    import klip_tpe.cli as cli
    print(f"klip_tpe_linked: linking {', '.join(link)} across partitions "
          f"(klip_tpe from {os.path.dirname(os.path.abspath(klip_tpe.__file__))})")
    cli.main(rest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
