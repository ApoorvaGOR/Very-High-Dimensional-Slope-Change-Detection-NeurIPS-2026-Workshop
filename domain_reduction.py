"""Sparse breakpoint domain reduction from Section 2 and Appendix A."""
import time

import numpy as np
from numba import njit

DEFAULT_REFINEMENTS = (1, 4, 8)


@njit
def _gain_statistics(x, Y):
    """Compute the baseline and affine hinge numerators at every knot."""
    J, T = Y.shape
    x_mean = x.mean()
    xc = x - x_mean
    x_ss = np.dot(xc, xc)
    means = np.empty(J)
    slopes = np.empty(J)
    baseline = np.zeros(J)
    numerator = np.zeros((T, J))
    derivative = np.empty((T, J))
    for j in range(J):
        means[j] = Y[j].mean()
        slopes[j] = np.dot(Y[j], xc) / x_ss
    suffix = np.zeros(J)
    for t in range(T - 1, -1, -1):
        for j in range(J):
            residual = Y[j, t] - means[j] - slopes[j] * xc[t]
            baseline[j] += residual * residual
            derivative[t, j] = -suffix[j]
            if t + 1 < T:
                numerator[t, j] = numerator[t + 1, j] + suffix[j] * (x[t + 1] - x[t])
            suffix[j] += residual
    return numerator, derivative, baseline, means, slopes, xc, x_ss


@njit
def _prefix_moments(x):
    """Stable group means and centered sums of squares, including repeated times."""
    means, ss = np.empty(len(x)), np.empty(len(x))
    mean = total = 0.0
    for i, value in enumerate(x):
        diff = value - mean
        mean += diff / (i + 1)
        total += diff * (value - mean)
        means[i], ss[i] = mean, total
    return means, ss


@njit
def _domain_coefficients(x, x_ss):
    left_mean, left_ss = _prefix_moments(x)
    right_mean, right_ss = _prefix_moments(x[::-1])
    T = len(x)
    coeffs = np.zeros((T - 1, 5))
    for t in range(1, T - 2):
        right_index = T - t - 2
        within_left, within_right = left_ss[t], right_ss[right_index]
        weight = (t + 1) * (right_index + 1) / T
        # Residual hinge norm via the within-left, within-right and
        # between-group components. This evaluates D as a sum of squares.
        coeffs[t, 0] = within_left * within_right / x_ss
        coeffs[t, 1] = weight * within_left / x_ss
        coeffs[t, 2] = weight * within_right / x_ss
        coeffs[t, 3] = right_mean[right_index] - x[t]
        coeffs[t, 4] = x[t] - left_mean[t]
    return coeffs


@njit(inline="always")
def _denominator(s, c0, c1, c2, right, left):
    return c0 + c1 * (right - s)**2 + c2 * (left + s)**2


@njit(inline="always")
def _gain(a, b, s, d, d2):
    if d > 0.:
        return (a + b * s)**2 / d
    # A repeated boundary time can make the hinge collinear at an endpoint.
    # Its one-sided limit still contributes to the interval upper gain bound.
    return b**2 / d2 if d2 > 0. else 0.


@njit
def _interval_gains(A, B, coeffs, lo, hi, midpoint_gains, maximum_gains):
    """Evaluate Section 2.2 in local coordinates s = z - x[t]."""
    c0, c1, c2, right, left = coeffs
    d0 = _denominator(0., c0, c1, c2, right, left)
    d1 = 2 * (c2 * left - c1 * right)
    d2 = c1 + c2
    mid = lo + (hi - lo) / 2
    dl = _denominator(lo, c0, c1, c2, right, left)
    du = _denominator(hi, c0, c1, c2, right, left)
    dm = _denominator(mid, c0, c1, c2, right, left)
    for j in range(len(A)):
        a, b = A[j], B[j]
        gm = (a + b * mid)**2 / dm if dm > 0. else 0.
        gmax = max(_gain(a, b, lo, dl, d2), _gain(a, b, hi, du, d2), gm)
        constant = 2 * b * d0 - a * d1
        linear = b * d1 - 2 * a * d2
        if linear != 0:
            root = -constant / linear
            if lo < root < hi:
                d = _denominator(root, c0, c1, c2, right, left)
                gmax = max(gmax, _gain(a, b, root, d, d2))
        midpoint_gains[j], maximum_gains[j] = gm, gmax


def _top_gain(gains, k):
    if k == 0:
        return 0.
    cutoff = len(gains) - k
    if cutoff:
        gains.partition(cutoff)
    return float(gains[cutoff:].sum())


@njit
def _recover_fit(x, Y, z, support, means, slopes, xc, x_ss):
    """Recover coefficients and evaluate residual SSE directly at the incumbent."""
    hinge = np.maximum(0., x - z)
    h_mean = hinge.mean()
    h_slope = np.dot(hinge, xc) / x_ss
    residual_hinge = hinge - h_mean - h_slope * xc
    d = np.dot(residual_hinge, residual_hinge)
    delta = np.zeros(len(Y))
    if d > 0.:
        for j in support:
            delta[j] = np.dot(Y[j], residual_hinge) / d
    beta = slopes - delta * h_slope
    intercept_at_mean = means - delta * h_mean
    alpha = intercept_at_mean - beta * x.mean()
    losses = np.zeros(len(Y))
    for j in range(len(Y)):
        for t in range(len(x)):
            residual = Y[j, t] - (intercept_at_mean[j] + beta[j] * xc[t] + delta[j] * hinge[t])
            losses[j] += residual * residual
    return alpha, beta, delta, losses


def domain_reduction_fit(x, Y, k, refinements):
    """Fit in normalized coordinates and report cumulative search checkpoints."""
    k = int(k)
    started = time.perf_counter()
    A, B, baseline, means, slopes, xc, x_ss = _gain_statistics(x, Y)
    coeffs = _domain_coefficients(x, x_ss)
    C = float(baseline.sum())
    statistics_runtime = time.perf_counter() - started
    search_started = time.perf_counter()
    intervals = [(t, 0., x[t + 1] - x[t]) for t in range(1, len(x) - 2) if x[t + 1] > x[t]]
    best_t, best_s, best_gain = intervals[0][0], intervals[0][2] / 2, -1.
    mid_gains, max_gains = np.empty(len(Y)), np.empty(len(Y))
    eps = 64 * np.finfo(float).eps
    objectives, lower_bounds, runtimes, retained_counts, evaluated_counts = [], [], [], [], []
    evaluated = 0
    for depth in range(refinements[-1] + 1):
        candidates = []
        for t, lo, hi in intervals:
            _interval_gains(A[t], B[t], coeffs[t], lo, hi, mid_gains, max_gains)
            gain = _top_gain(mid_gains, k)
            if gain > best_gain:
                best_gain, best_t, best_s = gain, t, lo + (hi - lo) / 2
            candidates.append((t, lo, hi, _top_gain(max_gains, k)))
        evaluated += len(intervals)
        # Keep near ties to avoid pruning on small floating-point differences.
        retained = [(t, lo, hi, bound) for t, lo, hi, bound in candidates
                    if bound > 0 and bound >= best_gain - eps * max(bound, best_gain)]
        if depth in refinements:
            gain_bound = max((bound for _, _, _, bound in retained), default=best_gain)
            gain_bound = max(gain_bound, best_gain)
            objectives.append(max(0., C - best_gain))
            lower_bounds.append(max(0., C - gain_bound - eps * max(C, gain_bound)))
            retained_counts.append(len(retained))
            evaluated_counts.append(evaluated)
            runtimes.append(statistics_runtime + time.perf_counter() - search_started)
        if depth < refinements[-1]:
            intervals = []
            for t, lo, hi, _ in retained:
                mid = lo + (hi - lo) / 2
                intervals.extend(((t, lo, mid), (t, mid, hi)))
    interval_runtime = time.perf_counter() - search_started

    recovery_started = time.perf_counter()
    z = float(x[best_t] + best_s)
    incumbent_scores = (A[best_t] + B[best_t] * best_s)**2
    support = (np.argpartition(incumbent_scores, len(Y) - k)[-k:]
               if k and best_gain > 0. else np.empty(0, dtype=np.int64))
    selected = np.zeros(len(Y), dtype=bool)
    selected[support] = True
    alpha, beta, delta, losses = _recover_fit(x, Y, z, support, means, slopes, xc, x_ss)
    recovery_runtime = time.perf_counter() - recovery_started
    return dict(k=k, C=C, Cj=baseline, shared_breakpoint=z, feasible_loss=float(losses.sum()),
                selected_support=support, selected_mask=selected, alpha=alpha, delta=delta,
                left_slope=beta, right_slope=beta + delta, left_intercept=alpha.copy(),
                right_intercept=alpha - delta * z, continuity_value=alpha + beta * z,
                losses_by_dimension=losses, refinement_iterations=np.array(refinements),
                refinement_feasible_SSE=np.array(objectives), refinement_lower_bound_SSE=np.array(lower_bounds),
                refinement_search_seconds=np.array(runtimes), refinement_intervals_retained=np.array(retained_counts),
                refinement_intervals_evaluated=np.array(evaluated_counts),
                statistics_runtime=statistics_runtime, interval_solve_runtime=interval_runtime,
                recovery_runtime=recovery_runtime, solve_runtime=statistics_runtime + interval_runtime + recovery_runtime)


def warm_up_domain_reduction():
    x = np.linspace(0., 1., 8)
    Y = np.vstack((x**2, np.maximum(0., x - .4), np.sin(x)))
    domain_reduction_fit(x, Y, 1, (1,))
