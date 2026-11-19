"""Reference implementation of the matrix projection used in the weight edit.

Directions are rows, matching the experiment's saved basis vectors. The output
basis stores directions as columns. The demo uses synthetic matrices.

Install NumPy, then run: python projection.py
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _matrix(name: str, values: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(values)
    if array.ndim != 2 or not all(array.shape):
        raise ValueError(f"{name} must be a nonempty matrix")
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return array.astype(np.float64, copy=False)


def orthonormal_basis(directions: ArrayLike) -> NDArray[np.float64]:
    """Return a column basis for the span of the supplied row directions."""
    rows = _matrix("directions", directions)
    vectors, singular_values, _ = np.linalg.svd(rows.T, full_matrices=False)
    tolerance = np.finfo(np.float64).eps * max(rows.shape) * singular_values[0]
    rank = np.count_nonzero(singular_values > tolerance)
    if rank == 0:
        raise ValueError("directions have zero rank")
    return vectors[:, :rank]


def project_out(
    weights: ArrayLike,
    basis: ArrayLike,
    *,
    strength: float = 1.0,
) -> NDArray[np.float64]:
    """Return W - strength * Q @ (Q.T @ W), with output coordinates on rows."""
    matrix = _matrix("weights", weights)
    columns = _matrix("basis", basis)
    if matrix.shape[0] != columns.shape[0]:
        raise ValueError("weights and basis must have the same output dimension")
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("strength must be between 0 and 1")
    gram = columns.T @ columns
    if not np.allclose(gram, np.eye(columns.shape[1]), rtol=0, atol=1e-10):
        raise ValueError("basis columns must be orthonormal")
    return matrix - strength * (columns @ (columns.T @ matrix))


def main() -> None:
    weights = np.array([[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]])
    directions = np.array([[1.0, 0.0, 1.0], [2.0, 0.0, 2.0]])
    basis = orthonormal_basis(directions)
    edited = project_out(weights, basis)
    print(f"Basis rank: {basis.shape[1]}")
    print(f"Component norm before: {np.linalg.norm(basis.T @ weights):.6g}")
    print(f"Component norm after:  {np.linalg.norm(basis.T @ edited):.6g}")


if __name__ == "__main__":
    main()
