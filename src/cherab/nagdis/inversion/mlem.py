"""Module for Maximum Likelihood Expectation Maximization (MLEM) algorithm."""

from __future__ import annotations

import numpy as np
from rich.console import Console
from rich.progress import Progress, TimeRemainingColumn
from scipy.sparse import spmatrix

__all__ = ["MLEM"]


class MLEM:
    """Maximum Likelihood Expectation Maximization (MLEM) algorithm.

    This class provides a simple implementation of the MLEM algorithm for solving the inverse problem
    :math:`\\mathbf{T} \\mathbf{x} = \\mathbf{b}` where :math:`\\mathbf{T}` is the forward problem matrix,
    :math:`\\mathbf{x}` is the unknown solution, and :math:`\\mathbf{b}` is the given data.

    Parameters
    ----------
    T : (M, N) ndarray | spmatrix
        Matrix :math:`\\mathbf{T}` of the forward problem.
    data : (M, ) or (M, K) array_like, optional
        Given data :math:`\\mathbf{b}_k`. :math:`k` time slices of the data vector
        :math:`\\begin{bmatrix} \\mathbf{b}_1 & \\mathbf{b}_2 & \\cdots & \\mathbf{b}_K \\end{bmatrix}`
        can be given as an array_like object.
    """

    def __init__(self, T, data) -> None:
        # validate arguments
        if not hasattr(T, "ndim"):
            raise TypeError("T must be an ndarray object")
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
        """Given data :math:`\\mathbf{b}_k`.

        :math:`k` time slices of the data vector
        :math:`\\begin{bmatrix} \\mathbf{b}_1 & \\mathbf{b}_2 & \\cdots & \\mathbf{b}_K \\end{bmatrix}`
        can be given as an array_like object.
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
        quiet: bool = False,
        store_temp: bool = False,
    ):
        """Solve the inverse problem using the MLEM algorithm.

        Parameters
        ----------
        x0 : (N, ) or (N, K) ndarray, optional
            Initial guess of the solution :math:`\\mathbf{x}`.
            If not given, a vector of ones is used.
        tol : float, optional
            Tolerance for convergence, by default 1e-5.
            The iteration stops when the maximum difference between the current and previous
            solutions is less than this value.
        max_iter : int, optional
            Maximum number of iterations, by default 100.
        quiet : bool, optional
            Whether to suppress the progress bar, by default False.
        store_temp : bool, optional
            Whether to store the temporary solutions during the iteration, by default False.

        Returns
        -------
        x : (N, ) or (N, K) ndarray
            Solution of the inverse problem.
            If the data has K time slices, the solution is a matrix.
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

        # set progress bar
        console = Console(quiet=quiet)
        progress = Progress(
            *Progress.get_default_columns()[:3],
            TimeRemainingColumn(elapsed_when_finished=True),
            auto_refresh=False,
            console=console,
        )
        task_id = progress.add_task("Pre-processing...", total=max_iter)
        progress.refresh()

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

        # start iteration
        with progress:
            # update progress bar
            progress.update(task_id, description="Solving...")
            progress.refresh()
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
                text = f"(Max Diff: {diff_max:.2e}, Tol: {_tol:.2e})"

                niter += 1
                progress.update(task_id, description=f"Solving...{text}", advance=1)
                progress.refresh()

            else:
                # stop progress bar
                progress.update(task_id, description=f"Completed {text}", completed=max_iter)
                progress.refresh()

        # set status
        status["elapsed_time"] = progress.tasks[task_id].elapsed
        status["niter"] = niter
        status["tol"] = _tol
        status["converged"] = self._converged
        status["diffs"] = diffs
        status["x_temp"] = x_temp

        return x, status
