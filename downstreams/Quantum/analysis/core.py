from typing import Any

import pandas as pd
import vector
import numpy as np
from itertools import product

from pandas import DataFrame
from scipy.linalg import sqrtm, eig

from downstreams.Quantum.analysis.basis import helicity_basis


def _get_particle_in_frame(particle_dict, frame: str) -> vector.Vector4D:
    if frame not in particle_dict:
        raise ValueError(f"Frame '{frame}' not available.")
    if particle_dict[frame] is None:
        raise ValueError(f"Particle not yet transformed to frame '{frame}'.")
    return particle_dict[frame]


def compute_full_density_matrix(C: dict[str, float], B: dict[str, float]) -> np.array:
    # Define Pauli matrices
    pauli_x = np.array([[0, 1], [1, 0]])  # σ1
    pauli_y = np.array([[0, -1j], [1j, 0]])  # σ2
    pauli_z = np.array([[1, 0], [0, -1]])  # σ3
    identity_2 = np.eye(2)

    # Define Kronecker product function for simplicity
    def kron(a, b):
        return np.kron(a, b)

    # Construct the density matrix ρ
    rho = (1 / 4) * (
            np.eye(4) +
            B['Bk'] * kron(pauli_x, identity_2) +
            B['Br'] * kron(pauli_y, identity_2) +
            B['Bn'] * kron(pauli_z, identity_2) +
            B['Ak'] * kron(identity_2, pauli_x) +
            B['Ar'] * kron(identity_2, pauli_y) +
            B['An'] * kron(identity_2, pauli_z) +
            C['kk'] * kron(pauli_x, pauli_x) +
            C['rr'] * kron(pauli_y, pauli_y) +
            C['nn'] * kron(pauli_z, pauli_z) +
            C['kr'] * kron(pauli_x, pauli_y) +
            C['kn'] * kron(pauli_x, pauli_z) +
            C['rn'] * kron(pauli_y, pauli_z) +
            C['rk'] * kron(pauli_y, pauli_x) +
            C['nr'] * kron(pauli_z, pauli_x) +
            C['nk'] * kron(pauli_z, pauli_y)
    )

    pauli_y_tensor = kron(pauli_y, pauli_y)
    rho_conjugate = np.conjugate(rho)
    rho_tilde = pauli_y_tensor @ rho_conjugate @ pauli_y_tensor
    # Compute the square root of ρ
    sqrt_rho = sqrtm(rho)
    # Compute the R matrix
    R = sqrt_rho @ rho_tilde @ sqrt_rho

    # Compute the eigenvalues of rho
    eigenvalues, _ = eig(R)
    eigenvalues = np.real(eigenvalues)
    eigenvalues.sort()
    eigenvalues = eigenvalues[::-1]

    return eigenvalues


def build_histograms(
        df: pd.DataFrame, bins: int = 100,
) -> dict[str, dict]:
    """
    Input:
        df: DataFrame with columns like 'cos_theta_A_n', etc.
        bins: number of bins
    Output:
        dict: key -> {'counts': array, 'edges': array, 'errors': array}
    """
    histograms = {}

    # 6 polarization terms: A_n, A_r, A_k, B_n, B_r, B_k
    for which, axis in product(['A', 'B'], ['n', 'r', 'k']):
        key = f'cos_theta_{which}_{axis}'
        values = df[key].values
        counts, edges = np.histogram(values, bins=bins, range=(-1, 1))
        errors = np.sqrt(counts)
        histograms[f'B_{which}{axis}'] = {
            'counts': counts,
            'edges': edges,
            'errors': errors
        }

    # 9 spin correlation terms: A_x * B_y for (n,r,k)
    for ax1, ax2 in product(['n', 'r', 'k'], repeat=2):
        keyA = f'cos_theta_A_{ax1}'
        keyB = f'cos_theta_B_{ax2}'
        product_values = df[keyA] * df[keyB]
        counts, edges = np.histogram(product_values, bins=bins, range=(-1, 1))
        errors = np.sqrt(counts)
        histograms[f'C_{ax1}{ax2}'] = {
            'counts': counts,
            'edges': edges,
            'errors': errors
        }

    return histograms


def calculate_B_C(
        histograms: dict[str, dict],
        kappas: tuple[float, float] = (1.0, -1.0)
) -> tuple[dict[Any, Any], dict[Any, Any], dict[Any, Any]]:
    result = {}
    result_up = {}
    result_down = {}

    def mean_and_error(counts, edges, errors):
        centers = 0.5 * (edges[:-1] + edges[1:])
        total = counts.sum()
        if total == 0:
            return 0.0, 0.0

        mean = np.sum(centers * counts) / total
        variance = np.sum(((centers - mean) / total) ** 2 * errors ** 2)
        return mean, np.sqrt(variance)

    # ✅ 1) B terms: for which, axis
    for which, axis in product(['A', 'B'], ['n', 'r', 'k']):
        key = f'B_{which}{axis}'
        h = histograms[key]
        mean, mean_err = mean_and_error(h['counts'], h['edges'], h['errors'])
        kappa = kappas[0] if which == 'A' else kappas[1]
        scaled_mean = mean * 3 / kappa
        scaled_err = mean_err * 3 / abs(kappa)
        result[key] = scaled_mean
        result_up[key] = scaled_mean + scaled_err
        result_down[key] = scaled_mean - scaled_err

    # ✅ 2) C terms: for ax1, ax2
    for ax1, ax2 in product(['n', 'r', 'k'], repeat=2):
        key = f'C_{ax1}{ax2}'
        h = histograms[key]
        mean, mean_err = mean_and_error(h['counts'], h['edges'], h['errors'])
        kappa_prod = kappas[0] * kappas[1]
        scaled_mean = mean * 9 / kappa_prod
        scaled_err = mean_err * 9 / abs(kappa_prod)
        result[key] = scaled_mean
        result_up[key] = scaled_mean + scaled_err
        result_down[key] = scaled_mean - scaled_err

    # return nominal, up, down values
    return result, result_up, result_down


def evaluate_quantum_results_with_uncertainties(results: dict, results_up: dict | None,
                                                results_down: dict | None) -> dict:
    """Compute eigenvalues and uncertainties using first-order perturbation theory with precomputed up/down values."""
    quantum_results = {}

    if results_up is None:
        results_up = results
    if results_down is None:
        results_down = results

    def compute_eigenvalues_from_results(input_results: dict) -> np.array:
        """Compute eigenvalues from a given set of results."""
        C = {key.replace("C_", ""): value for key, value in input_results.items() if 'C_' in key}
        B = {key.replace("B_", ""): value for key, value in input_results.items() if 'B_' in key}
        return compute_full_density_matrix(C=C, B=B)

    # Compute nominal eigenvalues
    nominal_eigenvalues = compute_eigenvalues_from_results(results)

    # Initialize uncertainty storage for asymmetric uncertainties
    eigenvalue_uncertainties_up = np.zeros_like(nominal_eigenvalues)
    eigenvalue_uncertainties_down = np.zeros_like(nominal_eigenvalues)

    # Compute uncertainties
    for param in results.keys():
        if param not in results_up or param not in results_down:
            continue  # Skip if no up/down value is provided for this parameter

        # Compute eigenvalues for up and down perturbed results
        eigenvalues_up = compute_eigenvalues_from_results({**results, param: results_up[param]})
        eigenvalues_down = compute_eigenvalues_from_results({**results, param: results_down[param]})

        # Accumulate squared uncertainties
        eigenvalue_uncertainties_up += np.abs(eigenvalues_up - nominal_eigenvalues) ** 2
        eigenvalue_uncertainties_down += np.abs(eigenvalues_down - nominal_eigenvalues) ** 2

    # Final uncertainties (square root of accumulated squares)
    eigenvalue_uncertainties_up = np.sqrt(eigenvalue_uncertainties_up)
    eigenvalue_uncertainties_down = np.sqrt(eigenvalue_uncertainties_down)

    # Compute Concurrence
    concurrence_nominal = max(0, nominal_eigenvalues[0] - sum(nominal_eigenvalues[1:]))
    concurrence_uncertainty_up = np.sqrt(sum(eigenvalue_uncertainties_up ** 2))
    concurrence_uncertainty_down = np.sqrt(sum(eigenvalue_uncertainties_down ** 2))

    quantum_results['Concurrence'] = {
        'value': concurrence_nominal,
        'uncertainty_up': concurrence_uncertainty_up,
        'uncertainty_down': concurrence_uncertainty_down
    }

    # Compute Cij terms with asymmetric uncertainties
    for i, j in [('kk', 'nn'), ('kk', 'rr'), ('nn', 'rr')]:
        value_sum = np.abs(results[f'C_{i}'] + results[f'C_{j}']) - np.sqrt(2)
        value_diff = np.abs(results[f'C_{i}'] - results[f'C_{j}']) - np.sqrt(2)

        uncertainty_up = np.sqrt(
            (results_up[f'C_{i}'] - results[f'C_{i}']) ** 2 + (results_up[f'C_{j}'] - results[f'C_{j}']) ** 2
        )
        uncertainty_down = np.sqrt(
            (results[f'C_{i}'] - results_down[f'C_{i}']) ** 2 + (results[f'C_{j}'] - results_down[f'C_{j}']) ** 2
        )

        quantum_results[f'C{i} + C{j}'] = {
            'value': value_sum,
            'uncertainty_up': uncertainty_up,
            'uncertainty_down': uncertainty_down
        }
        quantum_results[f'C{i} - C{j}'] = {
            'value': value_diff,
            'uncertainty_up': uncertainty_up,
            'uncertainty_down': uncertainty_down
        }

    return quantum_results


def build_results(truth_result, recon_result):
    for df in [truth_result, recon_result]:
        df['m_tt'] = df['mass']

        # 6 polarization terms: A_n, A_r, A_k, B_n, B_r, B_k
        for which, axis in product(['A', 'B'], ['n', 'r', 'k']):
            key = f'cos_theta_{which}_{axis}'
            df[f'B_{which}{axis}'] = df[key]
            # replace exactly zero values with NaN
            df[f'B_{which}{axis}'] = df[f'B_{which}{axis}'].replace(0, np.nan)

        # 9 spin correlation terms: A_x * B_y for (n,r,k)
        for ax1, ax2 in product(['n', 'r', 'k'], repeat=2):
            keyA = f'cos_theta_A_{ax1}'
            keyB = f'cos_theta_B_{ax2}'
            df[f'C_{ax1}{ax2}'] = df[keyA] * df[keyB]
            # replace exactly zero values with NaN
            df[f'C_{ax1}{ax2}'] = df[f'C_{ax1}{ax2}'].replace(0, np.nan)

    full_result = truth_result.merge(
        recon_result,
        left_index=True,
        right_index=True,
        suffixes=('_truth', '_recon')
    )

    # drop the original cos_theta columns
    for col in full_result.columns:
        if col.startswith('cos_theta_'):
            full_result.drop(columns=col, inplace=True)

    return full_result


class Core:
    def __init__(
            self,
            main_particle_1: vector.Vector4D, main_particle_2: vector.Vector4D,
            child1: vector.Vector4D, child2: vector.Vector4D,
    ):
        """
        Core class to calculate the cosine distribution between the parent particle and the children particles

        **All particles should be in the LAB Frame**
        :param main_particle_1: particle with positive charge to calculate the spin
        :param main_particle_2: particle with negative charge to calculate the spin
        :param child1: children particle of main_particle_1
        :param child2: children particle of main_particle_2
        """
        self._m1 = {
            'lab frame': main_particle_1,
        }
        self._m2 = {
            'lab frame': main_particle_2,
        }
        self._c1 = {
            'lab frame': child1,
        }
        self._c2 = {
            'lab frame': child2,
        }

        self._all_particles = {
            'm1': self._m1,
            'm2': self._m2,
            'c1': self._c1,
            'c2': self._c2,
        }

    def m1(self, frame: str) -> vector.Vector4D:
        return _get_particle_in_frame(self._m1, frame)

    def m2(self, frame: str) -> vector.Vector4D:
        return _get_particle_in_frame(self._m2, frame)

    def c1(self, frame: str) -> vector.Vector4D:
        return _get_particle_in_frame(self._c1, frame)

    def c2(self, frame: str) -> vector.Vector4D:
        return _get_particle_in_frame(self._c2, frame)

    def _transform_to_frame(self, boost: vector.Vector3D, source_frame: str, target_frame: str) -> None:
        for particle_dict in self._all_particles.values():
            particle_dict[target_frame] = particle_dict[source_frame].boost(boost)

    def analyze(self) -> DataFrame:
        # Transform to the center of mass frame
        m0 = self.m1('lab frame').add(self.m2('lab frame'))
        self._transform_to_frame(-m0.to_beta3(), 'lab frame', 'cm frame')

        # Transform to the rest frame of the parent particle
        self._transform_to_frame(-self.m1('cm frame').to_beta3(), 'cm frame', 'm1 cm frame')
        self._transform_to_frame(-self.m2('cm frame').to_beta3(), 'cm frame', 'm2 cm frame')

        # Calculate the helicity basis
        helicity_m1 = helicity_basis(self.m1('cm frame'))
        # helicity_m2 = helicity_basis(self.m2('cm frame'))

        # calculate cosine of the angle between the children particles and the helicity basis
        return pd.DataFrame({
            'mass': (self.m1('lab frame') + self.m2('lab frame')).mass,
            'theta_cm': 2 * np.arccos(np.abs(self.m1('cm frame').costheta)) / np.pi,
            'theta_cm_raw': np.arccos(self.m1('cm frame').costheta),

            'cos_theta_A_n': self.c1('m1 cm frame').to_pxpypz().unit().dot(helicity_m1['n']),
            'cos_theta_A_r': self.c1('m1 cm frame').to_pxpypz().unit().dot(helicity_m1['r']),
            'cos_theta_A_k': self.c1('m1 cm frame').to_pxpypz().unit().dot(helicity_m1['k']),

            # 'cos_theta_B_n': self.c2('m2 cm frame').to_pxpypz().unit().dot(helicity_m2['n']),
            # 'cos_theta_B_r': self.c2('m2 cm frame').to_pxpypz().unit().dot(helicity_m2['r']),
            # 'cos_theta_B_k': self.c2('m2 cm frame').to_pxpypz().unit().dot(helicity_m2['k']),

            'cos_theta_B_n': self.c2('m2 cm frame').to_pxpypz().unit().dot(helicity_m1['n']),
            'cos_theta_B_r': self.c2('m2 cm frame').to_pxpypz().unit().dot(helicity_m1['r']),
            'cos_theta_B_k': self.c2('m2 cm frame').to_pxpypz().unit().dot(helicity_m1['k']),
        })


if __name__ == "__main__":
    m1 = vector.Vector(x=44.9603, y=-0.893463, z=-505.204, t=507.204)
    m2 = vector.Vector(x=-44.9603, y=0.893463, z=-474.914, t=477.041)
    c1 = vector.Vector(x=2.25565, y=0.254229, z=-23.2934, t=23.4042)
    c2 = vector.Vector(x=-14.2044, y=-0.5068570, z=-147.73, t=148.412)

    core = Core(m1, m2, c1, c2)
    print(core.analyze())
