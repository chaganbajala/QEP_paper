"""Make the `QEP` core package (living at QEP_sparse/QEP/) importable from any module
in this package, without any dependency on Adaptive_bn/qep_shot.

afm_shots sits at QEP_sparse/QEP_multiple_shot/AFM_searching/afm_shots, i.e. two levels
below QEP_multiple_shot -- same resolution mechanism as qep_shot/_pathfix.py, just a
different relative depth (afm_shots -> AFM_searching -> QEP_multiple_shot -> QEP_sparse).

Usage: `from . import _pathfix` (for its side effect) before `from QEP import ...`.
"""
import os
import sys

_QEP_SPARSE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _QEP_SPARSE_DIR not in sys.path:
    sys.path.insert(0, _QEP_SPARSE_DIR)
