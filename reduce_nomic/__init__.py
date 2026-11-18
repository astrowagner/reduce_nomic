"""Python port of the IDL predecessor reduce_nomic.pro, the LBTI/NOMIC imaging reduction.

The package reduces one night of NOMIC raw frames to aligned, centered, frame-selected and
time-binned cubes for each chop position, with parallactic angles and median PSFs, in the
format KLIP-TPE reads. Stages: ``combine`` (chop states and coadds), ``chopsub`` (chop
subtraction and cosmetics), ``geometry`` (split, align, center), ``select`` (pre-clean, clean)
and ``binning`` (pre-binning, replacement), driven by ``pipeline.reduce``; ``idl`` holds the
IDL-compatible primitives and ``config`` the settings.
"""

__version__ = "0.1.0"
