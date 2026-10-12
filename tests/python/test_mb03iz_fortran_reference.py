"""Differential tests for MB03IZ against the SLICOT Fortran reference."""

from __future__ import annotations

import numpy as np
import pytest

from fortran_reference import run_fortran_driver


def _case(seed, m):
    rng = np.random.default_rng(seed)

    def cplx(shape):
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    a = np.triu(cplx((m, m)))
    c = np.tril(cplx((m, m)))
    d = cplx((m, m))
    b = np.triu(cplx((m, m)))
    f = np.triu(cplx((m, m)))
    f[np.diag_indices(m)] = f.diagonal().real
    return a, c, d, b, f


def _assign(name, mat):
    return "\n".join(
        f"  {name}({i + 1},{j + 1}) = ({v.real:.17e}, {v.imag:.17e})".replace("e", "d")
        for (i, j), v in np.ndenumerate(mat)
    )


def _driver(compq, compu, a, c, d, b, f):
    m = a.shape[0]
    mats = ["a", "c", "d", "b", "f", "q", "u1", "u2"]
    prints = "\n".join(
        f"  do j = 1, {2 * m if name == 'q' else m}\n"
        f"    do i = 1, {2 * m if name == 'q' else m}\n"
        f"      print '(2ES26.17)', dble({name}(i,j)), dimag({name}(i,j))\n"
        f"    end do\n"
        f"  end do"
        for name in mats
    )
    return f"""
program main
  implicit none
  integer, parameter :: m = {m}, n = 2*m
  complex*16 :: a(m,m), c(m,m), d(m,m), b(m,m), f(m,m), q(n,n), u1(m,m), u2(m,m)
  integer :: neig, info, i, j

  q = (0d0, 0d0)
  u1 = (0d0, 0d0)
  u2 = (0d0, 0d0)
{_assign("a", a)}
{_assign("c", c)}
{_assign("d", d)}
{_assign("b", b)}
{_assign("f", f)}

  call MB03IZ('{compq}', '{compu}', n, a, m, c, m, d, m, b, m, f, m, q, n, &
       u1, m, u2, m, neig, 0d0, info)

  print '(I0,1X,I0)', info, neig
{prints}
end program main
"""


@pytest.mark.parametrize("compq,compu", [("I", "I"), ("N", "N")])
@pytest.mark.parametrize("seed", [3, 11, 2026])
def test_mb03iz_matches_fortran(tmp_path, seed, compq, compu):
    from ctrlsys import mb03iz

    m = 4
    a, c, d, b, f = _case(seed, m)

    output = run_fortran_driver(_driver(compq, compu, a, c, d, b, f), tmp_path)
    lines = [ln for ln in output.split("\n") if ln.strip()]
    info_f, neig_f = (int(x) for x in lines[0].split())
    vals = np.array([complex(*map(float, ln.split())) for ln in lines[1:]])
    sizes = [m * m] * 5 + [4 * m * m, m * m, m * m]
    ref = []
    offset = 0
    for size in sizes:
        k = int(np.sqrt(size))
        ref.append(vals[offset:offset + size].reshape((k, k), order="F"))
        offset += size

    out = mb03iz(
        compq, compu, 2 * m,
        *(np.asfortranarray(x, dtype=np.complex128) for x in (a, c, d, b, f)),
        0.0,
    )
    a_o, c_o, d_o, b_o, f_o, q, u1, u2, neig, info = out

    assert info == info_f
    assert neig == neig_f
    for got, want in zip((a_o, c_o, d_o, b_o, f_o), ref[:5]):
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10)
    if compq == "I":
        np.testing.assert_allclose(q, ref[5], rtol=1e-10, atol=1e-10)
    if compu == "I":
        np.testing.assert_allclose(u1, ref[6], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(u2, ref[7], rtol=1e-10, atol=1e-10)
