/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#ifndef PY_WRAPPERS_H
#define PY_WRAPPERS_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include "slicot.h"

/* AB family */
extern PyObject* py_ab01md(PyObject* self, PyObject* args);
extern PyObject* py_ab01nd(PyObject* self, PyObject* args);
extern PyObject* py_ab01od(PyObject* self, PyObject* args);
extern PyObject* py_ab04md(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_ab05md(PyObject* self, PyObject* args);
extern PyObject* py_ab05nd(PyObject* self, PyObject* args);
extern PyObject* py_ab05od(PyObject* self, PyObject* args);
extern PyObject* py_ab05pd(PyObject* self, PyObject* args);
extern PyObject* py_ab05qd(PyObject* self, PyObject* args);
extern PyObject* py_ab05sd(PyObject* self, PyObject* args);
extern PyObject* py_ab05rd(PyObject* self, PyObject* args);
extern PyObject* py_ab07md(PyObject* self, PyObject* args);
extern PyObject* py_ab07nd(PyObject* self, PyObject* args);
extern PyObject* py_ab08md(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_ab08mz(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_ab08nd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_ab08nw(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_ab08nx(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_ab08ny(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_ab08nz(PyObject* self, PyObject* args);
extern PyObject* py_ab09ad(PyObject* self, PyObject* args);
extern PyObject* py_ab09ax(PyObject* self, PyObject* args);
extern PyObject* py_ab09bd(PyObject* self, PyObject* args);
extern PyObject* py_ab09bx(PyObject* self, PyObject* args);
extern PyObject* py_ab09cd(PyObject* self, PyObject* args);
extern PyObject* py_ab09cx(PyObject* self, PyObject* args);
extern PyObject* py_ab09dd(PyObject* self, PyObject* args);
extern PyObject* py_ab09ed(PyObject* self, PyObject* args);
extern PyObject* py_ab09fd(PyObject* self, PyObject* args);
extern PyObject* py_ab09gd(PyObject* self, PyObject* args);
extern PyObject* py_ab09hd(PyObject* self, PyObject* args);
extern PyObject* py_ab09hx(PyObject* self, PyObject* args);
extern PyObject* py_ab09hy(PyObject* self, PyObject* args);
extern PyObject* py_ab09id(PyObject* self, PyObject* args);
extern PyObject* py_ab09ix(PyObject* self, PyObject* args);
extern PyObject* py_ab09iy(PyObject* self, PyObject* args);
extern PyObject* py_ab09jd(PyObject* self, PyObject* args);
extern PyObject* py_ab09jv(PyObject* self, PyObject* args);
extern PyObject* py_ab09jw(PyObject* self, PyObject* args);
extern PyObject* py_ab09jx(PyObject* self, PyObject* args);
extern PyObject* py_ab09kd(PyObject* self, PyObject* args);
extern PyObject* py_ab09kx(PyObject* self, PyObject* args);
extern PyObject* py_ab09md(PyObject* self, PyObject* args);
extern PyObject* py_ab09nd(PyObject* self, PyObject* args);
extern PyObject* py_ab13ad(PyObject* self, PyObject* args);
extern PyObject* py_ab13bd(PyObject* self, PyObject* args);
extern PyObject* py_ab13cd(PyObject* self, PyObject* args);
extern PyObject* py_ab13dd(PyObject* self, PyObject* args);
extern PyObject* py_ab13dx(PyObject* self, PyObject* args);
extern PyObject* py_ab13ed(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_ab13fd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_ab13hd(PyObject* self, PyObject* args);
extern PyObject* py_ab13id(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_ab13md(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_ab8nxz(PyObject* self, PyObject* args);

/* AG family */
extern PyObject* py_ag07bd(PyObject* self, PyObject* args);
extern PyObject* py_ag08bd(PyObject* self, PyObject* args);
extern PyObject* py_ag08by(PyObject* self, PyObject* args);
extern PyObject* py_ag08bz(PyObject* self, PyObject* args);
extern PyObject* py_ag8byz(PyObject* self, PyObject* args);

/* BB family */
extern PyObject* py_bb01ad(PyObject* self, PyObject* args);
extern PyObject* py_bb02ad(PyObject* self, PyObject* args);
extern PyObject* py_bb03ad(PyObject* self, PyObject* args);
extern PyObject* py_bb04ad(PyObject* self, PyObject* args);

/* BD family */
extern PyObject* py_bd01ad(PyObject* self, PyObject* args);
extern PyObject* py_bd02ad(PyObject* self, PyObject* args);

/* DE family */
extern PyObject* py_de01od(PyObject* self, PyObject* args);
extern PyObject* py_de01pd(PyObject* self, PyObject* args);

/* DF family */
extern PyObject* py_df01md(PyObject* self, PyObject* args);

/* DG family */
extern PyObject* py_dg01md(PyObject* self, PyObject* args);
extern PyObject* py_dg01nd(PyObject* self, PyObject* args);
extern PyObject* py_dg01od(PyObject* self, PyObject* args);

/* DK family */
extern PyObject* py_dk01md(PyObject* self, PyObject* args);

/* FB family */
extern PyObject* py_fb01qd(PyObject* self, PyObject* args);
extern PyObject* py_fb01rd(PyObject* self, PyObject* args);
extern PyObject* py_fb01sd(PyObject* self, PyObject* args);
extern PyObject* py_fb01td(PyObject* self, PyObject* args);
extern PyObject* py_fb01vd(PyObject* self, PyObject* args);

/* FD family */
extern PyObject* py_fd01ad(PyObject* self, PyObject* args);

/* IB family */
extern PyObject* py_ib01ad(PyObject* self, PyObject* args);
extern PyObject* py_ib01bd(PyObject* self, PyObject* args);
extern PyObject* py_ib01cd(PyObject* self, PyObject* args);
extern PyObject* py_ib01md(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_ib01nd(PyObject* self, PyObject* args);
extern PyObject* py_ib01od(PyObject* self, PyObject* args);
extern PyObject* py_ib01oy(PyObject* self, PyObject* args);
extern PyObject* py_ib01pd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_ib01qd(PyObject* self, PyObject* args);
extern PyObject* py_ib01rd(PyObject* self, PyObject* args);
extern PyObject* py_ib03ad(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_ib03bd(PyObject* self, PyObject* args, PyObject* kwargs);

/* MA family */
extern PyObject* py_ma01ad(PyObject* self, PyObject* args);
extern PyObject* py_ma01bd(PyObject* self, PyObject* args);
extern PyObject* py_ma01bz(PyObject* self, PyObject* args);
extern PyObject* py_ma01cd(PyObject* self, PyObject* args);
extern PyObject* py_ma01dd(PyObject* self, PyObject* args);
extern PyObject* py_ma01dz(PyObject* self, PyObject* args);
extern PyObject* py_ma02ad(PyObject* self, PyObject* args);
extern PyObject* py_ma02bd(PyObject* self, PyObject* args);
extern PyObject* py_ma02cd(PyObject* self, PyObject* args);
extern PyObject* py_ma02dd(PyObject* self, PyObject* args);
extern PyObject* py_ma02ed(PyObject* self, PyObject* args);
extern PyObject* py_ma02es(PyObject* self, PyObject* args);
extern PyObject* py_ma02gd(PyObject* self, PyObject* args);
extern PyObject* py_ma02pd(PyObject* self, PyObject* args);
extern PyObject* py_ma02az(PyObject* self, PyObject* args);
extern PyObject* py_ma02bz(PyObject* self, PyObject* args);
extern PyObject* py_ma02cz(PyObject* self, PyObject* args);
extern PyObject* py_ma02ez(PyObject* self, PyObject* args);
extern PyObject* py_ma02gz(PyObject* self, PyObject* args);
extern PyObject* py_ma02hd(PyObject* self, PyObject* args);
extern PyObject* py_ma02hz(PyObject* self, PyObject* args);
extern PyObject* py_ma02iz(PyObject* self, PyObject* args);
extern PyObject* py_ma02jd(PyObject* self, PyObject* args);
extern PyObject* py_ma02jz(PyObject* self, PyObject* args);
extern PyObject* py_ma02md(PyObject* self, PyObject* args);
extern PyObject* py_ma02mz(PyObject* self, PyObject* args);
extern PyObject* py_ma02nz(PyObject* self, PyObject* args);
extern PyObject* py_ma02od(PyObject* self, PyObject* args);
extern PyObject* py_ma02oz(PyObject* self, PyObject* args);
extern PyObject* py_ma02pz(PyObject* self, PyObject* args);
extern PyObject* py_ma02rd(PyObject* self, PyObject* args);
extern PyObject* py_ma02sd(PyObject* self, PyObject* args);

/* MB01 family */
extern PyObject* py_mb01kd(PyObject* self, PyObject* args);
extern PyObject* py_mb01ld(PyObject* self, PyObject* args);
extern PyObject* py_mb01md(PyObject* self, PyObject* args);
extern PyObject* py_mb01nd(PyObject* self, PyObject* args);
extern PyObject* py_mb01oc(PyObject* self, PyObject* args);
extern PyObject* py_mb01od(PyObject* self, PyObject* args);
extern PyObject* py_mb01oe(PyObject* self, PyObject* args);
extern PyObject* py_mb01oh(PyObject* self, PyObject* args);
extern PyObject* py_mb01oo(PyObject* self, PyObject* args);
extern PyObject* py_mb01rh(PyObject* self, PyObject* args);
extern PyObject* py_mb01os(PyObject* self, PyObject* args);
extern PyObject* py_mb01ot(PyObject* self, PyObject* args);
extern PyObject* py_mb01rt(PyObject* self, PyObject* args);
extern PyObject* py_mb01pd(PyObject* self, PyObject* args);
extern PyObject* py_mb01qd(PyObject* self, PyObject* args);
extern PyObject* py_mb01rb(PyObject* self, PyObject* args);
extern PyObject* py_mb01rd(PyObject* self, PyObject* args);
extern PyObject* py_mb01ru(PyObject* self, PyObject* args);
extern PyObject* py_mb01rw(PyObject* self, PyObject* args);
extern PyObject* py_mb01rx(PyObject* self, PyObject* args);
extern PyObject* py_mb01ry(PyObject* self, PyObject* args);
extern PyObject* py_mb01sd(PyObject* self, PyObject* args);
extern PyObject* py_mb01ss(PyObject* self, PyObject* args);
extern PyObject* py_mb01td(PyObject* self, PyObject* args);
extern PyObject* py_mb01ud(PyObject* self, PyObject* args);
extern PyObject* py_mb01uw(PyObject* self, PyObject* args);
extern PyObject* py_mb01ux(PyObject* self, PyObject* args);
extern PyObject* py_mb01uy(PyObject* self, PyObject* args);
extern PyObject* py_mb01uz(PyObject* self, PyObject* args);
extern PyObject* py_mb01xd(PyObject* self, PyObject* args);
extern PyObject* py_mb01xy(PyObject* self, PyObject* args);
extern PyObject* py_mb01yd(PyObject* self, PyObject* args);
extern PyObject* py_mb01zd(PyObject* self, PyObject* args);
extern PyObject* py_mb01wd(PyObject* self, PyObject* args, PyObject* kwargs);

/* MB02 family */
extern PyObject* py_mb02cd(PyObject* self, PyObject* args);
extern PyObject* py_mb02dd(PyObject* self, PyObject* args);
extern PyObject* py_mb02cu(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb02cv(PyObject* self, PyObject* args);
extern PyObject* py_mb02cx(PyObject* self, PyObject* args);
extern PyObject* py_mb02cy(PyObject* self, PyObject* args);
extern PyObject* py_mb02ed(PyObject* self, PyObject* args);
extern PyObject* py_mb02fd(PyObject* self, PyObject* args);
extern PyObject* py_mb02gd(PyObject* self, PyObject* args);
extern PyObject* py_mb02hd(PyObject* self, PyObject* args);
extern PyObject* py_mb02id(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb02jd(PyObject* self, PyObject* args);
extern PyObject* py_mb02jx(PyObject* self, PyObject* args);
extern PyObject* py_mb02kd(PyObject* self, PyObject* args);
extern PyObject* py_mb02md(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb02nd(PyObject* self, PyObject* args);
extern PyObject* py_mb02ny(PyObject* self, PyObject* args);
extern PyObject* py_mb02od(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb02pd(PyObject* self, PyObject* args);
extern PyObject* py_mb02qd(PyObject* self, PyObject* args);
extern PyObject* py_mb02rd(PyObject* self, PyObject* args);
extern PyObject* py_mb02rz(PyObject* self, PyObject* args);
extern PyObject* py_mb02sd(PyObject* self, PyObject* args);
extern PyObject* py_mb02td(PyObject* self, PyObject* args);
extern PyObject* py_mb02sz(PyObject* self, PyObject* args);
extern PyObject* py_mb02tz(PyObject* self, PyObject* args);
extern PyObject* py_mb02ud(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb02uu(PyObject* self, PyObject* args);
extern PyObject* py_mb02uv(PyObject* self, PyObject* args);
extern PyObject* py_mb02uw(PyObject* self, PyObject* args);
extern PyObject* py_mb02vd(PyObject* self, PyObject* args);
extern PyObject* py_mb02wd(PyObject* self, PyObject* args);
extern PyObject* py_mb02yd(PyObject* self, PyObject* args);
extern PyObject* py_mb02xd(PyObject* self, PyObject* args);

/* MB03 family */
extern PyObject* py_mb03ab(PyObject* self, PyObject* args);
extern PyObject* py_mb03ad(PyObject* self, PyObject* args);
extern PyObject* py_mb03ae(PyObject* self, PyObject* args);
extern PyObject* py_mb03ag(PyObject* self, PyObject* args);
extern PyObject* py_mb03ah(PyObject* self, PyObject* args);
extern PyObject* py_mb03ai(PyObject* self, PyObject* args);
extern PyObject* py_mb03ba(PyObject* self, PyObject* args);
extern PyObject* py_mb03bc(PyObject* self, PyObject* args);
extern PyObject* py_mb03bd(PyObject* self, PyObject* args);
extern PyObject* py_mb03be(PyObject* self, PyObject* args);
extern PyObject* py_mb03bf(PyObject* self, PyObject* args);
extern PyObject* py_mb03bg(PyObject* self, PyObject* args);
extern PyObject* py_mb03bz(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb03cd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb03cz(PyObject* self, PyObject* args);
extern PyObject* py_mb03dd(PyObject* self, PyObject* args);
extern PyObject* py_mb03dz(PyObject* self, PyObject* args);
extern PyObject* py_mb03ed(PyObject* self, PyObject* args);
extern PyObject* py_mb03gd(PyObject* self, PyObject* args);
extern PyObject* py_mb03gz(PyObject* self, PyObject* args);
extern PyObject* py_mb03hd(PyObject* self, PyObject* args);
extern PyObject* py_mb03hz(PyObject* self, PyObject* args);
extern PyObject* py_mb03id(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb03ka(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb03kb(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb03kc(PyObject* self, PyObject* args);
extern PyObject* py_mb03ke(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb03md(PyObject* self, PyObject* args);
extern PyObject* py_mb03od(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb03oy(PyObject* self, PyObject* args);
extern PyObject* py_mb03pd(PyObject* self, PyObject* args);
extern PyObject* py_mb03py(PyObject* self, PyObject* args);
extern PyObject* py_mb03qd(PyObject* self, PyObject* args);
extern PyObject* py_mb03qx(PyObject* self, PyObject* args);
extern PyObject* py_mb03qy(PyObject* self, PyObject* args);
extern PyObject* py_mb03rd(PyObject* self, PyObject* args);
extern PyObject* py_mb03rx(PyObject* self, PyObject* args);
extern PyObject* py_mb03ry(PyObject* self, PyObject* args);
extern PyObject* py_mb03ud(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb03vd(PyObject* self, PyObject* args);
extern PyObject* py_mb03vy(PyObject* self, PyObject* args);
extern PyObject* py_mb03wd(PyObject* self, PyObject* args);
extern PyObject* py_mb03xd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb03xp(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb03xu(PyObject* self, PyObject* args);
extern PyObject* py_mb03ya(PyObject* self, PyObject* args);
extern PyObject* py_mb03yd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb03yt(PyObject* self, PyObject* args);
extern PyObject* py_mb3oyz(PyObject* self, PyObject* args);
extern PyObject* py_mb3pyz(PyObject* self, PyObject* args);
extern PyObject* py_mb03qg(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb03qv(PyObject* self, PyObject* args);
extern PyObject* py_mb03qw(PyObject* self, PyObject* args);
extern PyObject* py_mb03iz(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb03jd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb03jp(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb03jz(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb3jzp(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb3lzp(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb03rw(PyObject* self, PyObject* args);
extern PyObject* py_mb03rz(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb03sd(PyObject* self, PyObject* args);
extern PyObject* py_mb03ts(PyObject* self, PyObject* args);
extern PyObject* py_mb03td(PyObject* self, PyObject* args);
extern PyObject* py_mb03vw(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb03wx(PyObject* self, PyObject* args);
extern PyObject* py_mb03fd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb03fz(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb03kd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb03ld(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb03lf(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb03lp(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb03lz(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb03xs(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb03xz(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb03wa(PyObject* self, PyObject* args);
extern PyObject* py_mb03za(PyObject* self, PyObject* args);
extern PyObject* py_mb03zd(PyObject* self, PyObject* args);

/* MB04 family */
extern PyObject* py_mb04az(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb04ad(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb04bd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb04bz(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb04bp(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb04cd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb04db(PyObject* self, PyObject* args);
extern PyObject* py_mb04dd(PyObject* self, PyObject* args);
extern PyObject* py_mb04di(PyObject* self, PyObject* args);
extern PyObject* py_mb04dl(PyObject* self, PyObject* args);
extern PyObject* py_mb04dp(PyObject* self, PyObject* args);
extern PyObject* py_mb04ds(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb04dy(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb04dz(PyObject* self, PyObject* args);
extern PyObject* py_mb04ed(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb04fd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb04fp(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb04gd(PyObject* self, PyObject* args);
extern PyObject* py_mb04hd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb04id(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb04iy(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb04iz(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb04jd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb04kd(PyObject* self, PyObject* args);
extern PyObject* py_mb04ld(PyObject* self, PyObject* args);
extern PyObject* py_mb04md(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb04pa(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb04pu(PyObject* self, PyObject* args);
extern PyObject* py_mb04pb(PyObject* self, PyObject* args);
extern PyObject* py_mb04nd(PyObject* self, PyObject* args);
extern PyObject* py_mb04ny(PyObject* self, PyObject* args);
extern PyObject* py_mb04od(PyObject* self, PyObject* args);
extern PyObject* py_mb04ow(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb04ox(PyObject* self, PyObject* args);
extern PyObject* py_mb04oy(PyObject* self, PyObject* args);
extern PyObject* py_mb04py(PyObject* self, PyObject* args);
extern PyObject* py_mb04qb(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb04qc(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb04qf(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb04qu(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb04qs(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb04rb(PyObject* self, PyObject* args);
extern PyObject* py_mb04rs(PyObject* self, PyObject* args);
extern PyObject* py_mb04rt(PyObject* self, PyObject* args);
extern PyObject* py_mb04ru(PyObject* self, PyObject* args);
extern PyObject* py_mb04rv(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb04rw(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb04tb(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb04ts(PyObject* self, PyObject* args);
extern PyObject* py_mb04tt(PyObject* self, PyObject* args);
extern PyObject* py_mb04tu(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb04tv(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb04tw(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb04ty(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb04tx(PyObject* self, PyObject* args, PyObject* kwds);
extern PyObject* py_mb04vd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb04vx(PyObject* self, PyObject* args, PyObject* kwds);
extern PyObject* py_mb04su(PyObject* self, PyObject* args);
extern PyObject* py_mb04ud(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb04wu(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb04wd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb04wp(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb04wr(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb04yd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb04xd(PyObject* self, PyObject* args);
extern PyObject* py_mb04xy(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb04yw(PyObject* self, PyObject* args);
extern PyObject* py_mb04rd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb04rz(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb04zd(PyObject* self, PyObject* args);
extern PyObject* py_mb4dbz(PyObject* self, PyObject* args);
extern PyObject* py_mb4dlz(PyObject* self, PyObject* args);
extern PyObject* py_mb4dpz(PyObject* self, PyObject* args);

/* MB05 family */
extern PyObject* py_mb05md(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb05my(PyObject* self, PyObject* args);
extern PyObject* py_mb05nd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mb05od(PyObject* self, PyObject* args);
extern PyObject* py_mb05oy(PyObject* self, PyObject* args);

/* MC family */
extern PyObject* py_mc01md(PyObject* self, PyObject* args);
extern PyObject* py_mc01nd(PyObject* self, PyObject* args);
extern PyObject* py_mc01pd(PyObject* self, PyObject* args);
extern PyObject* py_mc01sx(PyObject* self, PyObject* args);
extern PyObject* py_mc01sy(PyObject* self, PyObject* args);
extern PyObject* py_mc01td(PyObject* self, PyObject* args);
extern PyObject* py_mc01od(PyObject* self, PyObject* args);
extern PyObject* py_mc01py(PyObject* self, PyObject* args);
extern PyObject* py_mc01qd(PyObject* self, PyObject* args);
extern PyObject* py_mc01rd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mc01sw(PyObject* self, PyObject* args);
extern PyObject* py_mc01wd(PyObject* self, PyObject* args);
extern PyObject* py_mc01xd(PyObject* self, PyObject* args);
extern PyObject* py_mc03md(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_mc03nd(PyObject* self, PyObject* args);
extern PyObject* py_mc03nx(PyObject* self, PyObject* args);
extern PyObject* py_mc03ny(PyObject* self, PyObject* args);
extern PyObject* py_mc01sd(PyObject* self, PyObject* args);
extern PyObject* py_mc01vd(PyObject* self, PyObject* args);

/* MD family */
extern PyObject* py_md03ba(PyObject* self, PyObject* args);
extern PyObject* py_md03bb(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_md03bd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_md03bf(PyObject* self, PyObject* args);
extern PyObject* py_md03by(PyObject* self, PyObject* args, PyObject* kwargs);

/* NF family */
extern PyObject* py_nf01ay(PyObject* self, PyObject* args);
extern PyObject* py_nf01br(PyObject* self, PyObject* args);
extern PyObject* py_nf01bs(PyObject* self, PyObject* args);
extern PyObject* py_nf01by(PyObject* self, PyObject* args);

/* SB family */
extern PyObject* py_sb01bd(PyObject* self, PyObject* args);
extern PyObject* py_sb01bx(PyObject* self, PyObject* args);
extern PyObject* py_sb02cx(PyObject* self, PyObject* args);
extern PyObject* py_sb02mr(PyObject* self, PyObject* args);
extern PyObject* py_sb02ms(PyObject* self, PyObject* args);
extern PyObject* py_sb02mv(PyObject* self, PyObject* args);
extern PyObject* py_sb02mw(PyObject* self, PyObject* args);
extern PyObject* py_sb02ou(PyObject* self, PyObject* args);
extern PyObject* py_sb02ov(PyObject* self, PyObject* args);
extern PyObject* py_sb02ow(PyObject* self, PyObject* args);
extern PyObject* py_sb02ox(PyObject* self, PyObject* args);
extern PyObject* py_sb01by(PyObject* self, PyObject* args);
extern PyObject* py_sb01dd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_sb01fy(PyObject* self, PyObject* args);
extern PyObject* py_sb01md(PyObject* self, PyObject* args);
extern PyObject* py_sb02md(PyObject* self, PyObject* args);
extern PyObject* py_sb02mt(PyObject* self, PyObject* args);
extern PyObject* py_sb02mu(PyObject* self, PyObject* args);
extern PyObject* py_sb02mx(PyObject* self, PyObject* args);
extern PyObject* py_sb02nd(PyObject* self, PyObject* args);
extern PyObject* py_sb02od(PyObject* self, PyObject* args);
extern PyObject* py_sb02oy(PyObject* self, PyObject* args);
extern PyObject* py_sb02pd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_sb02qd(PyObject* self, PyObject* args);
extern PyObject* py_sb02rd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_sb02ru(PyObject* self, PyObject* args);
extern PyObject* py_sb02sd(PyObject* self, PyObject* args);
extern PyObject* py_sb03md(PyObject* self, PyObject* args);
extern PyObject* py_sb03mv(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_sb03mw(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_sb03mx(PyObject* self, PyObject* args);
extern PyObject* py_sb03my(PyObject* self, PyObject* args);
extern PyObject* py_sb03mu(PyObject* self, PyObject* args);
extern PyObject* py_sb03od(PyObject* self, PyObject* args);
extern PyObject* py_sb03pd(PyObject* self, PyObject* args);
extern PyObject* py_sb03ou(PyObject* self, PyObject* args);
extern PyObject* py_sb03ov(PyObject* self, PyObject* args);
extern PyObject* py_sb03sx(PyObject* self, PyObject* args);
extern PyObject* py_sb03sy(PyObject* self, PyObject* args);
extern PyObject* py_sb03ud(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_sb03td(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_sb03qd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_sb03qx(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_sb03qy(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_sb03rd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_sb03sd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_sb04md(PyObject* self, PyObject* args);
extern PyObject* py_sb04mr(PyObject* self, PyObject* args);
extern PyObject* py_sb04mw(PyObject* self, PyObject* args);
extern PyObject* py_sb04nd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_sb04nv(PyObject* self, PyObject* args);
extern PyObject* py_sb04nw(PyObject* self, PyObject* args);
extern PyObject* py_sb04nx(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_sb04ny(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_sb04od(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_sb04ow(PyObject* self, PyObject* args);
extern PyObject* py_sb04pd(PyObject* self, PyObject* args);
extern PyObject* py_sb04py(PyObject* self, PyObject* args);
extern PyObject* py_sb04qd(PyObject* self, PyObject* args);
extern PyObject* py_sb04qr(PyObject* self, PyObject* args);
extern PyObject* py_sb04rv(PyObject* self, PyObject* args);
extern PyObject* py_sb04rw(PyObject* self, PyObject* args);
extern PyObject* py_sb04rx(PyObject* self, PyObject* args);
extern PyObject* py_sb04ry(PyObject* self, PyObject* args);
extern PyObject* py_sb04rd(PyObject* self, PyObject* args);
extern PyObject* py_sb08cd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_sb08dd(PyObject* self, PyObject* args);
extern PyObject* py_sb08ed(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_sb08fd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_sb08gd(PyObject* self, PyObject* args);
extern PyObject* py_sb08hd(PyObject* self, PyObject* args);
extern PyObject* py_sb08md(PyObject* self, PyObject* args);
extern PyObject* py_sb08my(PyObject* self, PyObject* args);
extern PyObject* py_sb08nd(PyObject* self, PyObject* args);
extern PyObject* py_sb08ny(PyObject* self, PyObject* args);
extern PyObject* py_sb09md(PyObject* self, PyObject* args);
extern PyObject* py_sb10dd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_sb10ed(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_sb10fd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_sb10hd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_sb10id(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_sb10jd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_sb10pd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_sb10ad(PyObject* self, PyObject* args);
extern PyObject* py_sb10ld(PyObject* self, PyObject* args);
extern PyObject* py_sb10rd(PyObject* self, PyObject* args);
extern PyObject* py_sb10sd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_sb10td(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_sb10vd(PyObject* self, PyObject* args);
extern PyObject* py_sb10wd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_sb10zd(PyObject* self, PyObject* args);
extern PyObject* py_sb10zp(PyObject* self, PyObject* args);
extern PyObject* py_sb10kd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* slicot_sb10yd(PyObject* self, PyObject* args);
extern PyObject* slicot_sb10md(PyObject* self, PyObject* args);
extern PyObject* py_sb06nd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_sb16ad(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_sb16ay(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_sb16bd(PyObject* self, PyObject* args);
extern PyObject* py_sb16cd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_sb16cy(PyObject* self, PyObject* args);
extern PyObject* py_sb04px(PyObject* self, PyObject* args);
extern PyObject* py_sb03oy(PyObject* self, PyObject* args);
extern PyObject* py_sb03or(PyObject* self, PyObject* args);
extern PyObject* py_sb03os(PyObject* self, PyObject* args);
extern PyObject* py_sb03ot(PyObject* self, PyObject* args);
extern PyObject* py_sb03oz(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_sb10ud(PyObject* self, PyObject* args, PyObject* kwargs);

/* SG family */
extern PyObject* py_sg02ad(PyObject* self, PyObject* args);
extern PyObject* py_sg02cv(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_sg03ad(PyObject* self, PyObject* args);
extern PyObject* py_sg03ax(PyObject* self, PyObject* args);
extern PyObject* py_sg03ay(PyObject* self, PyObject* args);
extern PyObject* py_sg03bd(PyObject* self, PyObject* args);
extern PyObject* py_sg03br(PyObject* self, PyObject* args);
extern PyObject* py_sg03bu(PyObject* self, PyObject* args);
extern PyObject* py_sg03bv(PyObject* self, PyObject* args);
extern PyObject* py_sg03bw(PyObject* self, PyObject* args);
extern PyObject* py_sg03bx(PyObject* self, PyObject* args);
extern PyObject* py_sg03bs(PyObject* self, PyObject* args);
extern PyObject* py_sg03bt(PyObject* self, PyObject* args);
extern PyObject* py_sg03by(PyObject* self, PyObject* args);
extern PyObject* py_sg03bz(PyObject* self, PyObject* args);

/* TB family */
extern PyObject* py_tb01id(PyObject* self, PyObject* args);
extern PyObject* py_tb01iz(PyObject* self, PyObject* args);
extern PyObject* py_tb01kd(PyObject* self, PyObject* args);
extern PyObject* py_tb01kx(PyObject* self, PyObject* args);
extern PyObject* py_tb01ld(PyObject* self, PyObject* args);
extern PyObject* py_tb01md(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_tb01nd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_tb01pd(PyObject* self, PyObject* args);
extern PyObject* py_tb01ud(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_tb01ux(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_tb01uy(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_tb01px(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_tb01td(PyObject* self, PyObject* args);
extern PyObject* py_tb01ty(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_tb01vd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_tb01vy(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_tb01wd(PyObject* self, PyObject* args);
extern PyObject* py_tb01wx(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_tb01xd(PyObject* self, PyObject* args);
extern PyObject* py_tb01xz(PyObject* self, PyObject* args);
extern PyObject* py_tb01zd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_tb03ad(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_tb03ay(PyObject* self, PyObject* args);
extern PyObject* py_tb04ad(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_tb04bv(PyObject* self, PyObject* args);
extern PyObject* py_tb04bd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_tb04cd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_tb04bw(PyObject* self, PyObject* args);
extern PyObject* py_tb04bx(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_tb05ad(PyObject* self, PyObject* args);

/* TC family */
extern PyObject* py_tc01od(PyObject* self, PyObject* args);
extern PyObject* py_tc04ad(PyObject* self, PyObject* args);
extern PyObject* py_tc05ad(PyObject* self, PyObject* args);

/* TD family */
extern PyObject* py_td03ad(PyObject* self, PyObject* args);
extern PyObject* py_td03ay(PyObject* self, PyObject* args);
extern PyObject* py_td04ad(PyObject* self, PyObject* args);
extern PyObject* py_td05ad(PyObject* self, PyObject* args);

/* TF family */
extern PyObject* py_tf01md(PyObject* self, PyObject* args);
extern PyObject* py_tf01mx(PyObject* self, PyObject* args);
extern PyObject* py_tf01my(PyObject* self, PyObject* args);
extern PyObject* py_tf01nd(PyObject* self, PyObject* args);
extern PyObject* py_tf01od(PyObject* self, PyObject* args);
extern PyObject* py_tf01pd(PyObject* self, PyObject* args);
extern PyObject* py_tf01qd(PyObject* self, PyObject* args);
extern PyObject* py_tf01rd(PyObject* self, PyObject* args);

/* TG family */
extern PyObject* py_tg01ad(PyObject* self, PyObject* args);
extern PyObject* py_tg01az(PyObject* self, PyObject* args);
extern PyObject* py_tg01bd(PyObject* self, PyObject* args);
extern PyObject* py_tg01cd(PyObject* self, PyObject* args);
extern PyObject* py_tg01dd(PyObject* self, PyObject* args);
extern PyObject* py_tg01ed(PyObject* self, PyObject* args);
extern PyObject* py_tg01fd(PyObject* self, PyObject* args);
extern PyObject* py_tg01fz(PyObject* self, PyObject* args);
extern PyObject* py_tg01gd(PyObject* self, PyObject* args);
extern PyObject* py_tg01hd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_tg01hu(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_tg01hx(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_tg01hy(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_tg01id(PyObject* self, PyObject* args);
extern PyObject* py_tg01jd(PyObject* self, PyObject* args);
extern PyObject* py_tg01jy(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_tg01kd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_tg01kz(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_tg01ld(PyObject* self, PyObject* args);
extern PyObject* py_tg01ly(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_tg01md(PyObject* self, PyObject* args);
extern PyObject* py_tg01nd(PyObject* self, PyObject* args);
extern PyObject* py_tg01nx(PyObject* self, PyObject* args);
extern PyObject* py_tg01oa(PyObject* self, PyObject* args);
extern PyObject* py_tg01ob(PyObject* self, PyObject* args);
extern PyObject* py_tg01od(PyObject* self, PyObject* args);
extern PyObject* py_tg01oz(PyObject* self, PyObject* args);
extern PyObject* py_tg01pd(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_tg01qd(PyObject* self, PyObject* args);
extern PyObject* py_tg01wd(PyObject* self, PyObject* args);

/* SG family wrappers */
extern PyObject* py_sg02cw(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_sg02cx(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_sg02nd(PyObject* self, PyObject* args, PyObject* kwargs);

/* UD family */
extern PyObject* py_ud01bd(PyObject* self, PyObject* args);
extern PyObject* py_ud01cd(PyObject* self, PyObject* args);
extern PyObject* py_ud01dd(PyObject* self, PyObject* args);
extern PyObject* py_ud01md(PyObject* self, PyObject* args);
extern PyObject* py_ud01mz(PyObject* self, PyObject* args, PyObject* kwargs);
extern PyObject* py_ud01nd(PyObject* self, PyObject* args, PyObject* kwargs);

/* UE family */
extern PyObject* py_ue01md(PyObject* self, PyObject* args);

/* LAPACK auxiliary routines (deprecated, not in modern LAPACK) */
extern PyObject* py_dlatzm(PyObject* self, PyObject* args);
extern PyObject* py_dgegv(PyObject* self, PyObject* args);
extern PyObject* py_zgegs(PyObject* self, PyObject* args);
extern PyObject* py_zgegv(PyObject* self, PyObject* args);

#endif /* PY_WRAPPERS_H */
