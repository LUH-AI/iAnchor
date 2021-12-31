from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class KL_LUCB:
    """
    Multi armed bandit with lower and upper bound.
    Used to find the anchor rule with the highest precision.
    """

    eps: float = field()
    delta: float = field()
    batch_size: int = field()
    verbose: bool = field()

    # TODO: fix type annotations
    def get_best_candidates(samples: list, top_n: int = 1):
        ...

