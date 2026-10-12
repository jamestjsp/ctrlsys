"""Differential tests for MB03FZ (and MB03IZ reordering) against the SLICOT Fortran reference."""

from __future__ import annotations

import numpy as np
import pytest

from fortran_reference import run_fortran_driver


HTML_Z = np.array([
    [0.0328 + 0.9611j, 0.6428 + 0.2585j, 0.7033 + 0.4254j, 0.2552 + 0.7053j],
    [0.0501 + 0.2510j, 0.2827 + 0.8865j, 0.4719 + 0.5387j, 0.0389 + 0.5676j],
    [0.5551 + 0.4242j, 0.0643 + 0.2716j, 0.1165 + 0.7875j, 0.9144 + 0.3891j],
    [0.0539 + 0.7931j, 0.0408 + 0.2654j, 0.9912 + 0.0989j, 0.0991 + 0.6585j],
])
HTML_B = np.array([
    [0.0547 + 0.8726j, 0.4008 + 0.8722j],
    [0.7423 + 0.6166j, 0.2631 + 0.5872j],
])
HTML_FG = np.array([
    [0.8740 + 0.0j, 0.3697 + 0.0j, 0.9178 + 0.6418j],
    [0.7748 + 0.5358j, 0.1652 + 0.0j, 0.2441 + 0.0j],
])


def _random_case(seed, n):
    rng = np.random.default_rng(seed)
    m = n // 2
    z = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    b = rng.standard_normal((m, m)) + 1j * rng.standard_normal((m, m))
    fg = rng.standard_normal((m, m + 1)) + 1j * rng.standard_normal((m, m + 1))
    for j in range(m):
        fg[j, j] = fg[j, j].real
        fg[j, j + 1] = fg[j, j + 1].real
    return z, b, fg


def _assign(name, mat):
    lines = []
    for (i, j), v in np.ndenumerate(mat):
        lines.append(f"  {name}({i + 1},{j + 1}) = ({v.real:.17e}, {v.imag:.17e})")
    return "\n".join(lines).replace("e", "d")


def _driver(compq, compu, orth, z, b, fg):
    n = z.shape[0]
    return f"""
program main
  implicit none
  integer, parameter :: n = {n}
  integer, parameter :: liwork = 2*n + 9, ldwork = 20*n*n + 200, lzwork = 20*n + 200
  complex*16 :: z(n,n), b(n,n), fg(n,n+1), d(n,n), c(n,n), q(2*n,2*n), u(n,2*n)
  complex*16 :: zwork(lzwork)
  double precision :: alphar(n), alphai(n), beta(n), dwork(ldwork)
  integer :: iwork(liwork), neig, info, i, j
  logical :: bwork(n)

  z = (0d0, 0d0)
  b = (0d0, 0d0)
  fg = (0d0, 0d0)
{_assign("z", z)}
{_assign("b", b)}
{_assign("fg", fg)}

  call MB03FZ('{compq}', '{compu}', '{orth}', n, z, n, b, n, fg, n, neig, d, n, c, n, &
       q, 2*n, u, n, alphar, alphai, beta, iwork, liwork, dwork, ldwork, &
       zwork, lzwork, bwork, info)

  print '(I0,1X,I0)', info, neig
  do i = 1, n
    print '(3ES26.17)', alphar(i), alphai(i), beta(i)
  end do
  do j = 1, neig
    do i = 1, n
      print '(2ES26.17)', dble(q(i,j)), dimag(q(i,j))
    end do
  end do
  do j = 1, neig
    do i = 1, n
      print '(2ES26.17)', dble(u(i,j)), dimag(u(i,j))
    end do
  end do
end program main
"""


def _parse(output, n):
    lines = output.split("\n")
    info, neig = (int(x) for x in lines[0].split())
    eig = np.array([[float(x) for x in lines[1 + i].split()] for i in range(n)])
    rows = [complex(*map(float, ln.split())) for ln in lines[1 + n:] if ln.strip()]
    q = np.array(rows[:n * neig]).reshape((neig, n)).T
    u = np.array(rows[n * neig:2 * n * neig]).reshape((neig, n)).T
    return info, neig, eig, q, u


def _projector(basis):
    return basis @ basis.conj().T


@pytest.mark.parametrize("orth", ["P", "Q", "S"])
@pytest.mark.parametrize(
    "case",
    ["html", "random8", "random12", "random40"],
)
def test_mb03fz_subspaces_match_fortran(tmp_path, case, orth):
    from ctrlsys import mb03fz

    if case == "html":
        z, b, fg = HTML_Z, HTML_B, HTML_FG
    elif case == "random8":
        z, b, fg = _random_case(2026, 8)
    elif case == "random12":
        z, b, fg = _random_case(7, 12)
    else:
        # N > LAPACK block size, so the multi-panel blocked updates run.
        z, b, fg = _random_case(40, 40)
    n = z.shape[0]

    output = run_fortran_driver(_driver("C", "C", orth, z, b, fg), tmp_path)
    info_f, neig_f, eig_f, q_f, u_f = _parse(output, n)

    (_, _, _, neig, _, _, q, u, alphar, alphai, beta, info) = mb03fz(
        "C", "C", orth, n,
        np.asfortranarray(z, dtype=np.complex128),
        np.asfortranarray(b, dtype=np.complex128),
        np.asfortranarray(fg, dtype=np.complex128),
    )

    assert info == info_f
    assert neig == neig_f
    np.testing.assert_allclose(alphar, eig_f[:, 0], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(alphai, eig_f[:, 1], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(beta, eig_f[:, 2], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        _projector(q[:n, :neig]), _projector(q_f), rtol=1e-9, atol=1e-9
    )
    np.testing.assert_allclose(
        _projector(u[:n, :neig]), _projector(u_f), rtol=1e-9, atol=1e-9
    )
