"""Module for Maximum Likelihood Expectation Maximization (MLEM) algorithm."""

from __future__ import annotations

from time import time

import numpy as np
from scipy.sparse import spmatrix

from cherab.inversion.tools.spinner import DummySpinner, Spinner

__all__ = ["MLEM"]


class MLEM:
    """Maximum Likelihood Expectation Maximization (MLEM) algorithm.

    This class provides a simple implementation of the MLEM algorithm for solving the inverse problem
    :math:`\\mathbf{T} \\mathbf{x} = \\mathbf{b}` where :math:`\\mathbf{T}` is the forward problem matrix,
    :math:`\\mathbf{x}` is the unknown solution, and :math:`\\mathbf{b}` is the given data.

    Parameters
    ----------
    T : array-like
        Matrix :math:`\\mathbf{T}` of the forward problem.
    data : array-like, optional
        Given data :math:`\\mathbf{b}`. :math:`n` time slices of the data vector
        :math:`\\begin{bmatrix} \\mathbf{b}_1 & \\mathbf{b}_2 & \\cdots & \\mathbf{b}_n \\end{bmatrix}`
        can be given as an array-like object.
    """

    def __init__(self, T, data) -> None:
        # validate arguments
        if not hasattr(T, "ndim"):
            raise TypeError("T must be an array-like object")
        if T.ndim != 2:
            raise ValueError("T must be a 2D array")

        # set matrix attributes
        self._T = T

        # set data attribute
        if data is not None:
            self.data = data

    @property
    def T(self) -> np.ndarray | spmatrix:
        """Matrix :math:`\\mathbf{T}` of the forward problem."""
        return self._T

    @property
    def data(self) -> np.ndarray:
        """Given data :math:`\\mathbf{b}`.

        :math:`n` time slices of the data vector
        :math:`\\begin{bmatrix} \\mathbf{b}_1 & \\mathbf{b}_2 & \\cdots & \\mathbf{b}_n \\end{bmatrix}`
        can be given as an array-like object.
        """
        return self._data

    @data.setter
    def data(self, value):
        data = np.asarray(value, dtype=float)
        if data.ndim == 1:
            data = data.transpose()
            size = data.size
        elif data.ndim == 2:
            size = data.shape[0]
        else:
            raise ValueError("data must be a vector or a matrix")
        if size != self._T.shape[0]:
            raise ValueError("data size must be the same as the number of rows of geometry matrix")
        self._data = data

    def solve(
        self,
        x0: np.ndarray | None = None,
        tol: float = 1e-5,
        max_iter: int = 100,
        spinner: bool = True,
        store_temp: bool = False,
    ):
        """Solve the inverse problem using the MLEM algorithm.

        Parameters
        ----------
        x0 : array-like, optional
            Initial guess of the solution :math:`\\mathbf{x}`. If not given, a vector of ones is used.
        tol : float, optional
            Tolerance for convergence. The iteration stops when the maximum difference between the current and
            previous solutions is less than this value.
        max_iter : int, optional
            Maximum number of iterations.
        spinner : bool, optional
            If True, a spinner is shown during the iteration.
        store_temp : bool, optional
            If True, the temporary solutions are stored in the status dictionary.

        Returns
        -------
        x : array-like
            Solution of the inverse problem.
        status : dict
            Dictionary containing the status of the iteration.
        """
        if self._data is None:
            raise ValueError("data must be set before calling solve method")

        # set initial guess
        if x0 is None:
            if self._data.ndim == 2:
                x0 = np.ones((self._T.shape[1], self._data.shape[1]))
            else:
                x0 = np.ones(self._T.shape[1])
        elif isinstance(x0, np.ndarray):
            if x0.ndim == 1:
                size = x0.size
            elif x0.ndim == 2:
                size = x0.shape[0]
            else:
                raise ValueError("x0 must be a vector or a matrix.")
            if size != self._T.shape[1]:
                raise ValueError("x0 must have the same size as the rows of T")

        # set tolerance
        def _tolerance(x):
            return tol * np.amax(x)

        # set spinner
        if spinner:
            _spinner = Spinner
        else:
            _spinner = DummySpinner

        # set iteration counter and status
        niter = 0
        status = {}
        self._converged = False
        diffs = []
        x = None  # solution
        x_temp = []  # temporary solutions
        data = None  # projection of the solution (T @ x)
        T_t = self._T.T  # transpose of T
        T_t1_recip = 1.0 / (T_t @ np.ones_like(self._data))  # 1 / (T^T @ 1)

        # set timer
        start_time = time()

        # start iteration
        with _spinner(f"{niter:03}-th iteration", timer=True) as sp:
            while niter < max_iter and not self._converged:
                data = self._T @ x0  # projection of the solution
                ratio = self._data / data
                x = x0 * (T_t @ ratio * T_t1_recip)

                # store temporary solution
                x_temp.append(x) if store_temp else None

                # check convergence
                diff_max = np.amax(np.abs(x - x0))
                diffs.append(diff_max)
                _tol = _tolerance(x0)
                self._converged = bool(diff_max < _tol)

                # update solution
                x0 = x
                sp.text = f"{niter:03}-th iteration (Max Diff: {diff_max:.3e}, Tol: {_tol:.3e})"

                niter += 1

            sp.ok()
        elapsed_time = time() - start_time

        # set status
        status["elapsed_time"] = elapsed_time
        status["niter"] = niter
        status["tol"] = _tol
        status["converged"] = self._converged
        status["diffs"] = diffs
        status["x_temp"] = x_temp

        return x, status
