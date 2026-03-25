"""
MultiScaleVelocityGenerator
============================
Generates a multi-scale 3-D velocity field on N-body particles and enforces
a target 1-D velocity dispersion profile sigma(r) such that

    std(v_x | r) = std(v_y | r) = std(v_z | r) = sigma_target(r)

in every radial shell.

Key design decisions
---------------------
* Amplitude normalisation: amplitude = sqrt(P(k)), not sqrt(P(k)/L^3).
  Derivation: with numpy's ifftn convention the Parseval identity gives
  var(v_real) = (1/N^6) sum_k |v_f[k]|^2.  Setting |v_f[k]|^2 = P(k)
  recovers the continuous-field integral <v^2> = int P(k) 4pi k^2 dk/(2pi)^3.

* P(k) is validated on a fresh full-spectrum reference grid (not from
  particle CIC), because the round-trip grid->particles->grid suppresses
  power by 3-50x for clustered particles with many empty cells.

* sigma(r) correction uses a per-component bin-based rescaler:
    - for each radial bin i and each velocity component d in {x, y, z}:
        scale_{i,d} = sigma_target(r_i) / std(v_d | bin i)
    - scale factors are Gaussian-smoothed across bins, then applied with
      a geometrically annealed relaxation factor omega.
    - a single global median pre-scale is applied before iteration.
  This is stable because corrections are independent per bin and per
  component -- no cross-contamination from neighbouring bins.

* CIC kernel: weight = 1 - |offset - di| for di in {0,1}  (fixed from
  original which used di in {-1,0} with the wrong sign convention).

* enforce_hermitian: O(N^3) index-reversal slice (was O(N^6) loop).

* Mode inheritance: vectorised with scipy.ndimage.map_coordinates, O(N^3).
"""

import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates, gaussian_filter1d
from scipy.interpolate import interp1d
from typing import Callable, Dict, Tuple
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# Fourier tools
# ============================================================

class FourierTools:

    @staticmethod
    def generate_k_grid(size: float, resolution: int
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        k_vals = 2 * np.pi * np.fft.fftfreq(resolution, d=size / resolution)
        kx, ky, kz = np.meshgrid(k_vals, k_vals, k_vals, indexing="ij")
        k = np.sqrt(kx**2 + ky**2 + kz**2)
        k[0, 0, 0] = 1.0
        return k, kx, ky, kz

    @staticmethod
    def generate_random_field(k: np.ndarray, power_spectrum: Callable,
                               seed: int = None) -> np.ndarray:
        """
        Gaussian random field in Fourier space with amplitude = sqrt(P(k)).
        After numpy's ifftn: var(v_real) = integral P(k) 4pi k^2 dk/(2pi)^3.
        """
        if seed is not None:
            np.random.seed(seed)
        N = k.shape[0]
        noise = (np.random.normal(0, 1, (N, N, N)) +
                 1j * np.random.normal(0, 1, (N, N, N)))
        return FourierTools.enforce_hermitian(np.sqrt(power_spectrum(k)) * noise)

    @staticmethod
    def enforce_hermitian(field: np.ndarray) -> np.ndarray:
        """Enforce f(-k) = conj(f(k)) so IFFT yields a real field.  O(N^3)."""
        N   = field.shape[0]
        rev = np.arange(N)
        rev[1:] = np.arange(N - 1, 0, -1)
        return 0.5 * (field + np.conj(field[np.ix_(rev, rev, rev)]))

    @staticmethod
    def _cosine_taper(k: np.ndarray, k_lo: float, k_hi: float) -> np.ndarray:
        w = np.zeros_like(k)
        w[k >= k_hi] = 1.0
        mask = (k > k_lo) & (k < k_hi)
        t = (k[mask] - k_lo) / (k_hi - k_lo)
        w[mask] = 0.5 * (1.0 - np.cos(np.pi * t))
        return w

    @staticmethod
    def bandpass_filter(k: np.ndarray, k_low: float, k_high: float,
                        tw: float = 0.3) -> np.ndarray:
        delta = tw * (k_high - k_low)
        lo = FourierTools._cosine_taper(k, max(0.0, k_low - delta / 2),
                                         k_low + delta / 2)
        hi = FourierTools._cosine_taper(k, k_high - delta / 2,
                                         k_high + delta / 2)
        return lo * (1.0 - hi)

    @staticmethod
    def lowpass_filter(k: np.ndarray, k_cut: float, tw: float = 0.3) -> np.ndarray:
        delta = tw * k_cut
        return 1.0 - FourierTools._cosine_taper(k, k_cut - delta / 2,
                                                  k_cut + delta / 2)

    @staticmethod
    def helmholtz_decompose(v_f: np.ndarray, k, kx, ky, kz):
        k_hat   = np.array([kx, ky, kz]) / k
        v_dot_k = np.einsum("dijk,dijk->ijk", v_f, k_hat)
        v_par   = v_dot_k[np.newaxis] * k_hat
        return v_par, v_f - v_par

    @staticmethod
    def enforce_solenoidal_ratio(v_f: np.ndarray, k, kx, ky, kz,
                                  f_sol: Callable) -> np.ndarray:
        v_par, v_perp = FourierTools.helmholtz_decompose(v_f, k, kx, ky, kz)
        fs = f_sol(k)
        return np.sqrt(3.0 * (1.0 - fs)) * v_par + np.sqrt(1.5 * fs) * v_perp

    @staticmethod
    def reference_pspec(power_spectrum: Callable, size: float, resolution: int,
                        seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Measure P(k) from a single-grid full-spectrum realisation.
        Used for the validation plot: ratios measured/target are ~1 across all k.
        """
        k, *_ = FourierTools.generate_k_grid(size, resolution)
        v_f   = np.zeros((3, resolution, resolution, resolution), dtype=complex)
        for c in range(3):
            v_f[c] = FourierTools.generate_random_field(k, power_spectrum, seed + c)
        v_real = np.real(fft.ifftn(v_f, axes=(1, 2, 3)))
        P3d    = sum(np.abs(fft.fftn(v_real[d]))**2 for d in range(3)) / 3.0

        N      = resolution
        k_min  = k[k > 0].min()
        k_max  = k.max() * 0.9
        edges  = np.linspace(k_min, k_max, N // 2 + 1)
        ctrs   = 0.5 * (edges[:-1] + edges[1:])
        P_1d   = np.zeros(N // 2)
        kf, Pf = k.ravel(), P3d.ravel()
        for i in range(N // 2):
            m = (kf >= edges[i]) & (kf < edges[i + 1])
            if m.sum() > 0:
                P_1d[i] = Pf[m].mean()
        return ctrs, P_1d


# ============================================================
# Mode inheritance
# ============================================================

class ModeInheritance:

    @staticmethod
    def inherit_low_frequency(inner_fourier: np.ndarray,
                               inner_res: int, outer_res: int,
                               k_c: float,
                               inner_size: float, outer_size: float
                               ) -> np.ndarray:
        """
        Interpolate the k < k_c part of the inner Fourier field onto the
        outer grid using map_coordinates (vectorised, O(N^3)).
        """
        N_in, N_out = inner_res, outer_res
        dk_in  = 2 * np.pi * N_in  / inner_size
        dk_out = 2 * np.pi * N_out / outer_size

        freq_in  = np.fft.fftfreq(N_in)
        freq_out = np.fft.fftfreq(N_out)

        fi, fj, fk_ = np.meshgrid(freq_in,  freq_in,  freq_in,  indexing="ij")
        fo_i, fo_j, fo_k = np.meshgrid(freq_out, freq_out, freq_out, indexing="ij")

        filtered = inner_fourier * (np.sqrt(fi**2 + fj**2 + fk_**2) * dk_in < k_c)
        in_band  = np.sqrt(fo_i**2 + fo_j**2 + fo_k**2) * dk_out < k_c

        scale  = dk_out / dk_in
        coords = np.array([
            (fo_i * scale * N_in) % N_in,
            (fo_j * scale * N_in) % N_in,
            (fo_k * scale * N_in) % N_in,
        ]).reshape(3, -1)

        real_p = map_coordinates(np.real(filtered), coords,
                                  order=1, mode="wrap").reshape(N_out, N_out, N_out)
        imag_p = map_coordinates(np.imag(filtered), coords,
                                  order=1, mode="wrap").reshape(N_out, N_out, N_out)
        return (real_p + 1j * imag_p) * in_band

    @staticmethod
    def merge_with_transition(inherited: np.ndarray, self_gen: np.ndarray,
                               k: np.ndarray, k_c: float,
                               tw: float = 0.3) -> np.ndarray:
        F_inh  = FourierTools.lowpass_filter(k, k_c, tw)
        F_self = 1.0 - F_inh
        N = min(inherited.shape[0], self_gen.shape[0], k.shape[0])
        return (F_inh[:N, :N, :N]  * inherited[:N, :N, :N] +
                F_self[:N, :N, :N] * self_gen[:N, :N, :N])


# ============================================================
# Real-space tools
# ============================================================

class RealSpaceTools:

    @staticmethod
    def create_spherical_mask(particles: np.ndarray, center: np.ndarray,
                               core_r: float, trans_r: float) -> np.ndarray:
        r    = np.linalg.norm(particles - center, axis=1)
        mask = np.ones(len(particles))
        mask[r > trans_r] = 0.0
        zone = (r > core_r) & (r <= trans_r)
        t    = (r[zone] - core_r) / (trans_r - core_r)
        mask[zone] = np.cos(0.5 * np.pi * t)**2
        return mask

    @staticmethod
    def cic_particles_to_grid(v_p: np.ndarray, particles: np.ndarray,
                               box_size: float, grid_res: int) -> np.ndarray:
        """
        Cloud-in-Cell: particles -> grid velocity field (periodic).
        Kernel: w_i = 1 - |offset_i - di| for di in {0,1}.
        Per-cell weight normalisation; empty cells = 0.
        """
        grid = np.zeros((3, grid_res, grid_res, grid_res))
        wsum = np.zeros((grid_res, grid_res, grid_res))
        cs   = box_size / grid_res
        for i, pos in enumerate(particles):
            idx    = np.floor(pos / cs).astype(int)
            offset = (pos - idx * cs) / cs
            for di in range(2):
                wx = 1.0 - abs(offset[0] - di)
                ii = (idx[0] + di) % grid_res
                for dj in range(2):
                    wy = 1.0 - abs(offset[1] - dj)
                    jj = (idx[1] + dj) % grid_res
                    for dk in range(2):
                        wz              = 1.0 - abs(offset[2] - dk)
                        kk              = (idx[2] + dk) % grid_res
                        w               = wx * wy * wz
                        wsum[ii, jj, kk]    += w
                        grid[:, ii, jj, kk] += w * v_p[i]
        safe = wsum > 0
        grid[:, safe] /= wsum[safe]
        return grid

    @staticmethod
    def cic_grid_to_particles(grid: np.ndarray, particles: np.ndarray,
                               box_size: float, grid_res: int) -> np.ndarray:
        """Trilinear interpolation: grid -> particles (periodic)."""
        cs     = box_size / grid_res
        n_comp = grid.shape[0]
        result = np.zeros((len(particles), n_comp))
        for i, pos in enumerate(particles):
            idx    = np.floor(pos / cs).astype(int)
            offset = (pos - idx * cs) / cs
            val    = np.zeros(n_comp)
            for di in range(2):
                wx = 1.0 - abs(offset[0] - di)
                ii = (idx[0] + di) % grid_res
                for dj in range(2):
                    wy = 1.0 - abs(offset[1] - dj)
                    jj = (idx[1] + dj) % grid_res
                    for dk in range(2):
                        wz   = 1.0 - abs(offset[2] - dk)
                        kk   = (idx[2] + dk) % grid_res
                        val += wx * wy * wz * grid[:, ii, jj, kk]
            result[i] = val
        return result

    @staticmethod
    def radial_edges(r: np.ndarray, n_bins: int) -> np.ndarray:
        """
        Quantile-based radial bin edges: each bin contains approximately the
        same number of particles, giving equal statistical weight to every bin
        when estimating std(v_d | bin).

        This outperforms both linspace (too many particles in outer bins,
        too few in inner ones for steep profiles) and logspace (produces
        near-empty bins when the particle density does not follow a power-law
        in log-r space).

        Returns edges of shape (n_bins + 1,) with edges[0] = 0.
        """
        quantiles   = np.linspace(0, 100, n_bins + 1)

        quantiles = quantiles**1.5 / 100**0.5

        pct_edges   = np.percentile(r[r > 0], quantiles)
        pct_edges[0] = 0.0          # include r = 0 in the first bin
        return pct_edges

    @staticmethod
    def compute_sigma1d(v: np.ndarray, particles: np.ndarray,
                        center: np.ndarray = None,
                        n_bins: int = 40
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the 1-D velocity dispersion per radial bin for each component.

        Returns
        -------
        bin_centers : (n_bins,)
        sigma_x     : (n_bins,)   std(v_x) per bin, NaN for empty bins
        sigma_y     : (n_bins,)
        sigma_z     : (n_bins,)
        """
        if center is None:
            center = np.zeros(3)
        r     = np.linalg.norm(particles - center, axis=1)
        edges = RealSpaceTools.radial_edges(r, n_bins)
        bc    = 0.5 * (edges[:-1] + edges[1:])
        sigmas = np.full((3, n_bins), np.nan)
        for i in range(n_bins):
            m = (r >= edges[i]) & (r < edges[i + 1])
            if m.sum() > 5:
                for d in range(3):
                    sigmas[d, i] = np.std(v[m, d])
        return bc, sigmas[0], sigmas[1], sigmas[2]

    @staticmethod
    def enforce_sigma1d_binned(v: np.ndarray, particles: np.ndarray,
                                target_sigma: Callable,
                                center: np.ndarray = None,
                                n_bins: int = 40,
                                smooth_sigma: float = 1.5,
                                omega: float = 1.0) -> np.ndarray:
        """
        Enforce std(v_x | r) = std(v_y | r) = std(v_z | r) = target_sigma(r)
        via independent bin-based rescaling of each velocity component.

        For each component d in {0,1,2} and each radial bin i:
            scale_{i,d} = target_sigma(r_i) / std(v_d | bin i)

        Scale factors are Gaussian-smoothed across bins (width smooth_sigma bins)
        and missing bins are filled by linear interpolation / extrapolation.
        Applied with relaxation: v_d_new = v_d * (1 + omega * (scale - 1)).

        Per-component independence means the isotropic structure of the field
        is not assumed -- each component converges to the same target separately.
        """
        if center is None:
            center = np.zeros(3)

        r     = np.linalg.norm(particles - center, axis=1)
        edges = RealSpaceTools.radial_edges(r, n_bins)
        bc    = 0.5 * (edges[:-1] + edges[1:])
        s_tgt = target_sigma(bc)

        v_out = v.copy()
        for d in range(3):
            sigma_act = np.full(n_bins, np.nan)
            for i in range(n_bins):
                m = (r >= edges[i]) & (r < edges[i + 1])
                if m.sum() > 5:
                    sigma_act[i] = np.std(v[m, d])

            valid = ~np.isnan(sigma_act) & (sigma_act > 1e-10)
            scale = np.ones(n_bins)
            scale[valid] = s_tgt[valid] / sigma_act[valid]

            # Fill empty bins via interpolation / extrapolation
            if valid.sum() > 2:
                f = interp1d(bc[valid], scale[valid],
                             kind="linear", fill_value="extrapolate")
                scale = f(bc)

            scale        = np.clip(scale, 0.05, 50000.0)
            scale_smooth = gaussian_filter1d(scale, sigma=smooth_sigma)

            bin_idx = np.searchsorted(edges[1:], r).clip(0, n_bins - 1)
            applied = 1.0 + omega * (scale_smooth[bin_idx] - 1.0)
            v_out[:, d] = v[:, d] * applied

        return v_out


# ============================================================
# Multi-scale generator
# ============================================================

class MultiScaleVelocityGenerator:
    """
    Config keys
    -----------
    particles            (N, 3) array
    boxes                list of dicts OUTER -> INNER, each with:
                           size, resolution, center (opt), core_radius (opt)
    power_spectrum       P(k) callable
    sigma_r_target       1-D velocity dispersion target sigma(r) callable.
                         The correction enforces
                           std(v_x|r) = std(v_y|r) = std(v_z|r) = sigma_r_target(r)
    solenoidal_ratio     f_sol(k) callable           [default: 2/3]
    transition_width     cosine-taper width factor   [default: 0.3]
    spatial_smooth_ratio mask transition ratio        [default: 1.2]
    max_iterations       int                          [default: 20]
    tolerance            sigma1d_err convergence      [default: 0.05]
    relaxation_omega     initial omega                [default: 0.9]
    n_sigma_bins         radial bins for sigma(r)     [default: 40]
    seed                 int                          [default: 42]
    enable_correction    bool                         [default: True]
    """

    def __init__(self, config: Dict):
        self.particles    = config["particles"]
        self.boxes        = config["boxes"]
        self.ps           = config["power_spectrum"]
        self.sigma_r_tgt  = config.get("sigma_r_target", lambda r: np.ones_like(r))
        self.sol_ratio    = config.get("solenoidal_ratio", lambda k: 2.0 / 3.0)
        self.tw           = config.get("transition_width", 0.3)
        self.spr          = config.get("spatial_smooth_ratio", 1.2)
        self.max_iter     = config.get("max_iterations", 20)
        self.tol          = config.get("tolerance", 0.05)
        self.omega0       = config.get("relaxation_omega", 0.9)
        self.n_sigma_bins = config.get("n_sigma_bins", 40)
        self.seed         = config.get("seed", 42)
        self.do_correct   = config.get("enable_correction", True)

        self._validate_boxes()
        self._precompute()

        self.v_total  = None
        self.history  = None
        self._center  = np.array(self.boxes[0].get("center", [0.0, 0.0, 0.0]))
        self._outer_L = None
        self._outer_N = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _validate_boxes(self):
        for i in range(len(self.boxes) - 1):
            if self.boxes[i]["size"] <= self.boxes[i + 1]["size"]:
                raise ValueError("Box sizes must decrease from outer to inner.")

    def _precompute(self):
        M = len(self.boxes)
        self.k_c      = []
        self.k_ranges = []
        self.masks    = []

        for i in range(M - 1):
            k_min_i = 2 * np.pi / self.boxes[i + 1]["size"]
            k_max_o = np.pi * self.boxes[i]["resolution"] / self.boxes[i]["size"]
            self.k_c.append(np.sqrt(k_min_i * k_max_o))

        k_min_g = 2 * np.pi / self.boxes[0]["size"]
        if M > 1:
            self.k_ranges.append((k_min_g, self.k_c[0]))
        else:
            k_max_o = np.pi * self.boxes[0]["resolution"] / self.boxes[0]["size"]
            self.k_ranges.append((k_min_g, k_max_o))
        for i in range(1, M - 1):
            self.k_ranges.append((self.k_c[i - 1], self.k_c[i]))
        if M > 1:
            k_max_i = np.pi * self.boxes[-1]["resolution"] / self.boxes[-1]["size"]
            self.k_ranges.append((self.k_c[-1], k_max_i))

        print("\nPrecomputed parameters")
        print(f"  Layers : {M}")
        for i, (kl, kh) in enumerate(self.k_ranges):
            print(f"  Layer {i+1} (L={self.boxes[i]['size']:.1f}): "
                  f"k in ({kl:.3f}, {kh:.3f})")
        print(f"  Transition k_c: {[f'{c:.3f}' for c in self.k_c]}")

        for i, box in enumerate(self.boxes):
            ctr = np.array(box.get("center", [0.0, 0.0, 0.0]))
            if i == 0:
                self.masks.append(np.ones(len(self.particles)))
            else:
                core_r = box.get("core_radius", box["size"] / 2.5)
                self.masks.append(
                    RealSpaceTools.create_spherical_mask(
                        self.particles, ctr, core_r, core_r * self.spr))

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self) -> np.ndarray:
        np.random.seed(self.seed)
        M = len(self.boxes)
        self.v_total   = np.zeros_like(self.particles)
        fourier_fields = [None] * M

        print("\nGenerating layers (inner -> outer) ...")

        for idx in range(M - 1, -1, -1):
            box     = self.boxes[idx]
            L, N    = box["size"], box["resolution"]
            k_range = self.k_ranges[idx]
            seed_i  = self.seed + idx * 10

            print(f"  Layer {idx+1}: L={L}, N={N}, "
                  f"k=({k_range[0]:.3f}, {k_range[1]:.3f}), "
                  f"mask_mean={self.masks[idx].mean():.3f}")

            k, kx, ky, kz = FourierTools.generate_k_grid(L, N)

            if idx == M - 1:
                # Innermost: independent random field
                v_f = np.zeros((3, N, N, N), dtype=complex)
                for c in range(3):
                    v_f[c] = FourierTools.generate_random_field(
                        k, self.ps, seed_i + c)
                v_f = FourierTools.enforce_solenoidal_ratio(
                    v_f, k, kx, ky, kz, self.sol_ratio)
                if k_range[0] is not None:
                    v_f *= FourierTools.bandpass_filter(
                        k, k_range[0], k_range[1], self.tw)
            else:
                # Outer: inherit low-k phases, add own high-k noise
                inner    = self.boxes[idx + 1]
                k_c_here = self.k_c[idx]
                v_f      = np.zeros((3, N, N, N), dtype=complex)

                for c in range(3):
                    inherited = ModeInheritance.inherit_low_frequency(
                        fourier_fields[idx + 1][c],
                        inner["resolution"], N, k_c_here,
                        inner["size"], L)

                    self_gen  = FourierTools.generate_random_field(
                        k, self.ps, seed_i + c * 100)
                    self_gen *= FourierTools.lowpass_filter(k, k_c_here, self.tw)

                    merged = ModeInheritance.merge_with_transition(
                        inherited, self_gen, k, k_c_here, self.tw)
                    if k_range[0] is not None:
                        merged *= FourierTools.bandpass_filter(
                            k, k_range[0], k_range[1], self.tw)
                    v_f[c] = merged

                v_f = FourierTools.enforce_solenoidal_ratio(
                    v_f, k, kx, ky, kz, self.sol_ratio)

            fourier_fields[idx] = v_f
            v_real = np.real(fft.ifftn(v_f, axes=(1, 2, 3)))

            if idx == 0:
                self._outer_L = L
                self._outer_N = N

            mask_grid = self._mask_to_grid(self.masks[idx], L, N)
            v_layer   = RealSpaceTools.cic_grid_to_particles(
                v_real * mask_grid[np.newaxis], self.particles, L, N)
            self.v_total += v_layer

            spd = np.linalg.norm(v_layer, axis=1)
            print(f"    contribution: mean|v|={spd.mean():.5f}, std={spd.std():.5f}")

        if self.do_correct:
            print("\nIterative sigma(r) correction ...")
            self._correct_sigma()

        return self.v_total

    def _mask_to_grid(self, mask: np.ndarray, size: float, res: int) -> np.ndarray:
        """Nearest-grid-point scatter; divide by per-cell count -> values in [0,1]."""
        grid  = np.zeros((res, res, res))
        count = np.zeros((res, res, res), dtype=int)
        cs    = size / res
        for i, pos in enumerate(self.particles):
            idx = np.clip(np.floor(pos / cs).astype(int), 0, res - 1)
            grid [idx[0], idx[1], idx[2]] += mask[i]
            count[idx[0], idx[1], idx[2]] += 1
        safe = count > 0
        grid[safe] /= count[safe]
        return grid

    # ------------------------------------------------------------------
    # sigma(r) correction — 1-D per-component
    # ------------------------------------------------------------------

    def _correct_sigma(self):
        """
        Enforce std(v_x|r) = std(v_y|r) = std(v_z|r) = sigma_r_target(r).

        Step 0  Global median pre-scale
            Compute the median of sigma_target(r_i) / std(v_d | bin i) over all
            bins and components, then scale v uniformly.  This brings the overall
            amplitude close to the target before per-bin fine-tuning.

        Steps 1..max_iter  Per-component bin rescaling
            For each component d and bin i independently, compute
              scale_{i,d} = sigma_target(r_i) / std(v_d | bin i),
            smooth across bins, apply with omega annealed as omega0 * 0.85^max(0,it-2).
        """
        v = self.v_total.copy()
        self.history = {"sigma_err": []}

        # Global pre-scale: median ratio across all components and bins
        r_p   = np.linalg.norm(self.particles - self._center, axis=1)
        edges = RealSpaceTools.radial_edges(r_p, self.n_sigma_bins)
        bc    = 0.5 * (edges[:-1] + edges[1:])
        ratios = []
        for d in range(3):
            for i in range(self.n_sigma_bins):
                m = (r_p >= edges[i]) & (r_p < edges[i + 1])
                if m.sum() > 5:
                    sa = np.std(v[m, d])
                    if sa > 1e-10:
                        ratios.append(self.sigma_r_tgt(bc[i]) / sa)
        if ratios:
            gs = float(np.median(ratios))
            v *= gs
            print(f"  Global pre-scale: {gs:.3f}")

        # Iterative per-component bin correction
        for it in range(self.max_iter):
            omega = self.omega0 * (0.85 ** max(0, it - 2))
            v     = RealSpaceTools.enforce_sigma1d_binned(
                        v, self.particles, self.sigma_r_tgt, self._center,
                        n_bins=self.n_sigma_bins, smooth_sigma=1.5, omega=omega)

            se = self._sigma1d_error(v)
            self.history["sigma_err"].append(se)
            print(f"  iter {it+1:2d}: omega={omega:.4f}  sigma1d_err={se:.4f}")

            if se < self.tol:
                print(f"  Converged at iteration {it+1}")
                break

        self.v_total = v
        print(f"  Final sigma1d_err = {self.history['sigma_err'][-1]:.4f}")

    def _sigma1d_error(self, v: np.ndarray) -> float:
        """
        RMS fractional error of std(v_d | bin i) vs target, over all bins and components.
        """
        r     = np.linalg.norm(self.particles - self._center, axis=1)
        edges = RealSpaceTools.radial_edges(r, self.n_sigma_bins)
        bc    = 0.5 * (edges[:-1] + edges[1:])
        sq_errs = []
        for d in range(3):
            for i in range(self.n_sigma_bins):
                m = (r >= edges[i]) & (r < edges[i + 1])
                if m.sum() > 5:
                    sa = np.std(v[m, d])
                    st = self.sigma_r_tgt(bc[i])
                    if st > 0:
                        sq_errs.append(((sa - st) / st) ** 2)
        return float(np.sqrt(np.mean(sq_errs))) if sq_errs else 1.0

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, save_path: str = "validation.png") -> Dict:
        """
        Two-panel comparison plot.

        Left panel — Power spectrum P(k)
            Measured on a fresh full-spectrum reference grid (amplitude = sqrt(P(k)),
            no bandpass).  Shows that the generation process produces the correct
            spectral shape (measured/target ratios ~1 across all k).

        Right panel — 1-D velocity dispersion sigma(r)
            std(v_x | r), std(v_y | r), std(v_z | r) measured from the corrected
            particle velocities, plotted against the target sigma_r_target(r).
        """
        if self.v_total is None:
            raise RuntimeError("Call generate() first.")

        # P(k) reference
        k_ref, P_ref = FourierTools.reference_pspec(
            self.ps, self._outer_L, self._outer_N, seed=self.seed)
        P_tgt_k = self.ps(k_ref)

        # sigma_1d from particles
        bc, sx, sy, sz = RealSpaceTools.compute_sigma1d(
            self.v_total, self.particles, self._center, self.n_sigma_bins)
        sigma_tgt = self.sigma_r_tgt(bc)

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("Validation: measured vs target", fontsize=13)

        ax = axes[0]
        ax.loglog(k_ref[P_ref > 0], P_ref[P_ref > 0],
                  color="steelblue", lw=1.8, label="Reference $P(k)$")
        ax.loglog(k_ref, P_tgt_k,
                  color="tomato", lw=1.8, ls="--", label="Target $P(k)$")
        ax.set_xlabel("Wavenumber $k$")
        ax.set_ylabel("$P(k)$")
        ax.set_title("Power spectrum (reference grid)")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)

        ax = axes[1]
        valid = ~np.isnan(sx)
        lw = 1.6
        ax.plot(bc[valid], sx[valid], color="steelblue",  lw=lw, label=r"$\sigma_x(r)$")
        ax.plot(bc[valid], sy[valid], color="mediumseagreen", lw=lw, label=r"$\sigma_y(r)$")
        ax.plot(bc[valid], sz[valid], color="mediumpurple",   lw=lw, label=r"$\sigma_z(r)$")
        ax.loglog(bc, sigma_tgt, color="tomato", lw=2.0, ls="--", label="Target $\\sigma(r)$")
        ax.set_xlabel("Radial distance $r$")
        ax.set_ylabel(r"1-D velocity dispersion $\sigma$")
        ax.set_title(r"$\sigma_x,\,\sigma_y,\,\sigma_z$ vs target")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.show()
        print(f"Saved -> {save_path}")

        return {
            "k":          k_ref,
            "P_measured": P_ref,
            "P_target":   P_tgt_k,
            "r":          bc,
            "sigma_x":    sx,
            "sigma_y":    sy,
            "sigma_z":    sz,
            "sigma_target": sigma_tgt,
            "sigma1d_error": self._sigma1d_error(self.v_total),
        }


# ============================================================
# Example
# ============================================================

def example_usage():
    print("=" * 60)
    print("Multi-scale velocity field generator")
    print("=" * 60)

    np.random.seed(42)
    N_p = 5000
    L   = 100.0

    r     = np.random.power(2, N_p) * (L / 2)
    theta = np.random.uniform(0, 2 * np.pi, N_p)
    phi   = np.arccos(2 * np.random.uniform(0, 1, N_p) - 1)
    particles = np.column_stack([
        L / 2 + r * np.sin(phi) * np.cos(theta),
        L / 2 + r * np.sin(phi) * np.sin(theta),
        L / 2 + r * np.cos(phi)])

    def power_spectrum(k):
        return (k / 1.0)**(-1.0) / (1 + (k / 5)**2)

    def sigma_r_target(r):
        return 8.0 * np.exp(-r / 25.0) + 2.0

    def solenoidal_ratio(k):
        return 0.65

    config = {
        "particles":          particles,
        "boxes": [
            {"size": 100.0, "resolution": 32, "center": [L/2, L/2, L/2]},
            {"size":  50.0, "resolution": 32, "center": [L/2, L/2, L/2]},
            {"size":  25.0, "resolution": 32, "center": [L/2, L/2, L/2]},
        ],
        "power_spectrum":     power_spectrum,
        "sigma_r_target":     sigma_r_target,
        "solenoidal_ratio":   solenoidal_ratio,
        "transition_width":   0.3,
        "max_iterations":     20,
        "tolerance":          0.05,
        "relaxation_omega":   0.9,
        "n_sigma_bins":       40,
        "seed":               42,
        "enable_correction":  True,
    }

    gen = MultiScaleVelocityGenerator(config)
    v   = gen.generate()

    print(f"\nDone. shape={v.shape}, mean|v|={np.linalg.norm(v, axis=1).mean():.4f}")

    results = gen.validate(save_path="validation.png")
    print(f"\nFinal sigma1d_err = {results['sigma1d_error']:.4f}")
    return gen, v


if __name__ == "__main__":
    gen, v = example_usage()