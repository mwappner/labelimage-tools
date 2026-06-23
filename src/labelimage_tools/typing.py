from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

Node = int | np.integer
Adj = Mapping[Node, Iterable[Node]]
Neig = dict[Node, np.ndarray]
Cont = dict[Node, np.ndarray]
