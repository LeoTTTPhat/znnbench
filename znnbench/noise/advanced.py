"""Additional additive noise models for noise-tolerant ZNN studies.

Every factory returns ``callable(t, rng, shape) -> ndarray`` with ``shape =
(B, m)``.  As in ``noise/basic.py`` the noise is injected in the design-
formula space, J x_dot = model + nu(t), which is how the noise-tolerant ZNN
literature perturbs the error dynamics  E_dot = -gamma Phi(E) + nu(t)
(e.g. Liao et al. 2022, Front. Neurorobot., eq. (6); Liao et al. 2022,
Mathematics, eq. (8)).  Full citations and equations: docs/spec/noise_info.md.

Coefficients may be scalars (every entry of the noise matrix is the same,
the convention ``[a]_{n x n}`` used in the cited papers) or arrays that
broadcast against ``shape`` (e.g. a per-entry coefficient vector of length m).

Keys registered here
--------------------
``uniform``         bounded random noise U[low, high]^{B x m}, i.i.d. per call
                    (Liao, Hua, Cao, Katsikis, Li 2022, Remark 3: [-0.5, 0.5];
                    Yan et al. 2019, IEEE Access: [7.5, 8.5]).
``quadratic``       a t^2 + b t + c  (Liao et al. 2022, W_3(t) = A t^2 + B t + C).
``polynomial``      sum_k c_k t^k   (generalisation named in Liao et al. 2022).
``multi_harmonic``  sum_k Gamma_k sin(2 pi f_k t + phi_k)
                    (Liao, Wang, Li, Guo, He 2022, Front. Neurorobot., eq. (12)).
``combined``        constant + linear + harmonic (+ optional bounded random)
                    superposition.  ASSUMPTION: benchmark composite of the
                    literature components (cf. Yang et al. 2024, arXiv
                    2412.20477, who use sin t + cos t + bounded white noise).
``random_walk``     Brownian-like random walk  sigma * W(t).  BENCHMARK
                    EXTENSION: no ZNN paper using it was located.
``impulse``         impulse/burst noise (Poisson or periodic bursts of fixed
                    width).  BENCHMARK EXTENSION: no ZNN paper using it was
                    located.

Stateful noises (``random_walk``, ``impulse``) remember their last call; a call
with a time earlier than the previous one (a new simulation starting again at
t = 0) or with a different shape resets them.  Times within ~1e-11 s of the
previous call count as the same instant (round-off of RK4 stage times).
"""
from __future__ import annotations

import numpy as np

from ..registry import register_noise


def _coef(c):
    return np.asarray(c, dtype=float)


def _earlier(t, t_prev):
    """True if t is genuinely earlier than t_prev (a new run), ignoring
    round-off duplicates such as an RK4 stage time t_k + tau vs (k+1) tau."""
    return t < t_prev - 1e-11 * max(1.0, abs(t_prev))


def _full(value, shape):
    """Broadcast a scalar / per-entry coefficient array to the noise shape."""
    return np.array(np.broadcast_to(np.asarray(value, dtype=float), shape), dtype=float)


# ---------------------------------------------------------------------------
# bounded random noise
# ---------------------------------------------------------------------------
@register_noise("uniform")
def uniform(low=-0.5, high=0.5):
    """Bounded random noise with i.i.d. entries ~ U[low, high] at every call.

    Liao, Hua, Cao, Katsikis, Li (2022), Mathematics 10(15):2817, Remark 3:
    "bounded random noises [-0.5, 0.5]^{2x2}" (the default).  Yan, Xiao, Li,
    Zhang, Yan, Liu (2019), IEEE Access 7:41517, eq. (6) and Figs. 2-3, use
    the non-zero-mean random noise [7.5, 8.5]^{2x2}  ->  low=7.5, high=8.5.
    ASSUMPTION: a fresh draw per vector-field evaluation; the papers give only
    the bounds, not the distribution or the sampling rate -> uniform.
    """
    low, high = float(low), float(high)
    if high < low:
        raise ValueError("uniform noise needs low <= high")
    return lambda t, rng, shape: rng.uniform(low, high, size=shape)


# ---------------------------------------------------------------------------
# polynomial-in-time noises
# ---------------------------------------------------------------------------
@register_noise("polynomial")
def polynomial(coeffs=(0.0, 0.0, 1.0)):
    """nu(t) = sum_k c_k t^k, coefficients in ascending powers.

    Liao et al. (2022, Mathematics) note that other nonlinear noises can be
    approximated by polynomial noises (Taylor expansion), quadratic noise
    being one kind.  Each c_k may be a scalar or broadcast to (B, m).
    """
    cs = [_coef(c) for c in coeffs]
    if not cs:
        raise ValueError("polynomial noise needs at least one coefficient")

    def f(t, rng, shape):
        acc = np.zeros(shape)
        for c in reversed(cs):          # Horner
            acc = acc * t + c
        return acc
    return f


@register_noise("quadratic")
def quadratic(a=1.0, b=0.0, c=0.0):
    """Quadratic noise W_3(t) = A t^2 + B t + C (Liao et al. 2022, Mathematics,
    Sec. 3; real quadratic noise [t^2]_{2x2} in Sec. 4.3, complex [(0.5+0.5i)
    t^2] and [(5+5i) t^2] in Example 2).  Defaults reproduce [t^2]."""
    return polynomial((c, b, a))


# ---------------------------------------------------------------------------
# harmonic superpositions
# ---------------------------------------------------------------------------
@register_noise("multi_harmonic")
def multi_harmonic(amps=(1.0, 2.0, 5.0), freqs=(1.0, 2.0, 5.0), phases=(1.0, 2.0, 4.0)):
    """Multiple-harmonic noise  o~_ij(t) = sum_k Gamma_k sin(2 pi f_k t + phi_k)
    (Liao, Wang, Li, Guo, He 2022, Front. Neurorobot. 16:928636, eq. (12));
    frequencies in Hz.  Defaults: the periodic example of Sec. 5,
    sin(2 pi t + 1) + 2 sin(4 pi t + 2) + 5 sin(10 pi t + 4).  The aperiodic
    example sin(1) + 2 sin(5 t + 2) + 5 sin(10 pi t + 3) is
    amps=(1, 2, 5), freqs=(0, 2.5/pi, 5), phases=(1, 2, 3); the single
    harmonics 10^3 sin(0.002 pi t + 1) and 10^3 sin(100 pi t + 1) are
    amps=[1e3], freqs=[0.001] or [50], phases=[1].
    Single-harmonic eq. (7) is the special case of one term (also available
    as ``harmonic`` in noise/basic.py with omega = 2 pi f)."""
    amps, freqs, phases = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (amps, freqs, phases))
    if not (len(amps) == len(freqs) == len(phases)):
        raise ValueError("amps, freqs and phases must have equal length")

    def f(t, rng, shape):
        val = float(np.sum(amps * np.sin(2.0 * np.pi * freqs * t + phases)))
        return np.full(shape, val)
    return f


@register_noise("combined")
def combined(level=1.0, slope=0.1, amp=1.0, omega=1.0, phase=0.0, rand_bound=0.0):
    """Composite noise  nu(t) = level + slope t + amp sin(omega t + phase)
    [+ U(-rand_bound, rand_bound)].

    ASSUMPTION (benchmark composite): superposition of the constant, linear,
    harmonic (and bounded random) noises of the cited noise-tolerant ZNN
    papers.  The defaults are benchmark choices, not values from a paper.
    The composite of Yang, Wang, Voyles, Ma (arXiv:2412.20477, Sec. 4.2),
    sin t + cos t + n(t) with white noise |n| <= 2, is level=0, slope=0,
    amp=sqrt(2), omega=1, phase=pi/4, rand_bound=2 (ASSUMPTION: the white
    noise is drawn uniform; the paper gives only its bound).
    By linearity of the error dynamics, the steady-state residual is the sum
    of the per-component responses, which makes this a compact stress test
    of a model's simultaneous rejection capability.
    """
    level, slope, amp, omega, phase, rb = map(float, (level, slope, amp, omega, phase, rand_bound))

    def f(t, rng, shape):
        out = np.full(shape, level + slope * t + amp * np.sin(omega * t + phase))
        if rb > 0:
            out = out + rng.uniform(-rb, rb, size=shape)
        return out
    return f


# ---------------------------------------------------------------------------
# stateful stochastic noises
# ---------------------------------------------------------------------------
class RandomWalkNoise:
    """nu(t) = x0 + sigma W(t) with W a standard Wiener process sampled at the
    call times: nu(t') = nu(t) + sigma sqrt(t' - t) xi, xi ~ N(0, I).

    BENCHMARK EXTENSION (no ZNN source located).  The value is held when the
    same t is queried again (e.g. several RK4 stages); a call with t smaller
    than the previous one, or with a new shape, restarts the walk at x0.
    """

    def __init__(self, sigma=1.0, x0=0.0):
        self.sigma = float(sigma)
        self.x0 = x0
        self._t = None
        self._v = None

    def reset(self):
        self._t, self._v = None, None

    def __call__(self, t, rng, shape):
        t = float(t)
        if self._v is None or self._v.shape != tuple(shape) or _earlier(t, self._t):
            self._t, self._v = t, _full(self.x0, shape)
        elif t > self._t + 1e-11 * max(1.0, abs(self._t)):
            self._v = self._v + self.sigma * np.sqrt(t - self._t) * rng.standard_normal(shape)
            self._t = t
        return self._v.copy()


@register_noise("random_walk")
def random_walk(sigma=1.0, x0=0.0):
    return RandomWalkNoise(sigma=sigma, x0=x0)


class ImpulseNoise:
    """Impulse / burst noise: rectangular bursts of height ``amp`` (times a
    random sign per burst and entry if ``random_sign``) and duration ``width``.

    Burst start times are either periodic, ``phase + j * period``, or a Poisson
    process of intensity ``rate`` (bursts per second) whose arrivals are drawn
    lazily from the simulation rng.  Outside bursts the noise is zero.

    BENCHMARK EXTENSION (no ZNN source located).  With sampling period tau a
    burst is seen by a causal discretization only if some t_k lies inside it:
    choose width >= tau to guarantee every burst is sampled at least once.
    """

    def __init__(self, amp=10.0, rate=0.5, width=0.05, period=None, phase=0.0,
                 random_sign=True):
        if width <= 0:
            raise ValueError("impulse width must be positive")
        if period is None and rate <= 0:
            raise ValueError("Poisson impulse noise needs rate > 0")
        self.amp, self.rate, self.width = float(amp), float(rate), float(width)
        self.period = None if period is None else float(period)
        self.phase = float(phase)
        self.random_sign = bool(random_sign)
        self.reset()

    def reset(self):
        self._t = None
        self._t0 = None
        self._shape = None
        self._starts = []          # start times of the bursts still relevant
        self._signs = []           # sign pattern of each burst
        self._last_start = None    # last periodic start generated
        self._next = None          # next (not yet materialised) Poisson arrival

    def _sign(self, rng, shape):
        if self.random_sign:
            return np.where(rng.random(shape) < 0.5, -1.0, 1.0)
        return np.ones(shape)

    def _extend(self, t, rng, shape):
        """Materialise all bursts that start at or before t."""
        if self.period is not None:
            nxt = self.phase if self._last_start is None else self._last_start + self.period
            while nxt <= t:
                self._starts.append(nxt)
                self._signs.append(self._sign(rng, shape))
                self._last_start = nxt
                nxt += self.period
            return
        if self._next is None:
            self._next = self._t0 + rng.exponential(1.0 / self.rate)
        while self._next <= t:
            self._starts.append(self._next)
            self._signs.append(self._sign(rng, shape))
            self._next += rng.exponential(1.0 / self.rate)

    def __call__(self, t, rng, shape):
        t = float(t)
        shape = tuple(shape)
        if self._t is None or shape != self._shape or _earlier(t, self._t):
            self.reset()
            self._shape = shape
            self._t0 = t
        self._t = max(t, self._t) if self._t is not None else t
        self._extend(t, rng, shape)
        # drop bursts that ended before t (starts are increasing and t is
        # non-decreasing between resets, so they can never be active again)
        keep = [i for i, s in enumerate(self._starts) if t - s < self.width]
        self._starts = [self._starts[i] for i in keep]
        self._signs = [self._signs[i] for i in keep]
        out = np.zeros(shape)
        for sg in self._signs:     # active bursts (overlapping ones add up)
            out = out + self.amp * sg
        return out


@register_noise("impulse")
def impulse(amp=10.0, rate=0.5, width=0.05, period=None, phase=0.0, random_sign=True):
    return ImpulseNoise(amp=amp, rate=rate, width=width, period=period, phase=phase,
                        random_sign=random_sign)
