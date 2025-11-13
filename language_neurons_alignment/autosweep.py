from __future__ import annotations
from typing import List, Tuple, Optional

from .pipeline import run_identify_and_count

def _run_once(
    model_name: str,
    model_path: str,
    dataset: str,
    toprate: float,
    lam: float,
    activation_bar_ratio: float,
) -> List[int]:
    _, appear = run_identify_and_count(
        model_name=model_name,
        model_path=model_path,
        dataset=dataset,
        toprate=toprate,
        lam=lam,
        activation_bar_ratio=activation_bar_ratio,
    )
    return appear


def autosweep_lambda(
    model_name: str,
    model_path: str,
    dataset: str,
    toprate: float,
    lo_init: float = 0.0,
    hi_init: float = 0.2,
    eps: float = 1e-3,
    activation_bar_ratio: float = 0.95,
) -> Tuple[Optional[float], List[int]]:
    print(f"[autosweep] evaluate at lo={lo_init:.6f}")
    appear_lo = _run_once(model_name, model_path, dataset, toprate, lo_init, activation_bar_ratio)
    print(f"[autosweep] lambda={lo_init:.6f}, appear={appear_lo}")

    print(f"[autosweep] evaluate at hi={hi_init:.6f}")
    appear_hi = _run_once(model_name, model_path, dataset, toprate, hi_init, activation_bar_ratio)
    print(f"[autosweep] lambda={hi_init:.6f}, appear={appear_hi}")

    best_lambda: Optional[float] = None
    best_appear: List[int] = []

    def ok(appear: List[int]) -> bool:
        return len(appear) > 10 and appear[10] == 0

    if ok(appear_lo):
        best_lambda = lo_init
        best_appear = appear_lo
    if ok(appear_hi):
        best_lambda = hi_init
        best_appear = appear_hi

    lo, hi = lo_init, hi_init
    last_appear = appear_hi

    max_iters = 64
    it = 0
    while (hi - lo) > eps and it < max_iters:
        it += 1
        mid = (lo + hi) / 2.0
        appear_mid = _run_once(model_name, model_path, dataset, toprate, mid, activation_bar_ratio)
        print(f"[autosweep] iter={it:02d}  lambda={mid:.6f}, appear={appear_mid}")
        last_appear = appear_mid

        if ok(appear_mid):
            best_lambda = mid
            best_appear = appear_mid
            lo = mid
        else:
            hi = mid

    if best_lambda is None:
        print("[autosweep] No lambda in range makes appear[10]==0. "
              "Returning (None, last_appear).")
        return None, last_appear

    print(f"[autosweep] Best lambda (max with appear[10]==0): {best_lambda:.6f}")
    print(f"[autosweep] Best appear: {best_appear}")
    return best_lambda, best_appear
