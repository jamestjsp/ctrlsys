API Reference
=============

This library provides **578 functions** across 27 categories.

.. code-block:: python

   from slicot import ab01md, sb03md


AB - Analysis - Controllability, Observability, Minimal Realization
-------------------------------------------------------------------

System analysis: controllability/observability staircase forms, transmission zeros,
system interconnections (series/parallel/feedback), model reduction (balanced truncation,
Hankel-norm, frequency-weighted), and system norms (H2, H-infinity, Hankel).

.. autofunction:: slicot.ab01md
.. autofunction:: slicot.ab01nd
.. autofunction:: slicot.ab01od
.. autofunction:: slicot.ab04md
.. autofunction:: slicot.ab05md
.. autofunction:: slicot.ab05nd
.. autofunction:: slicot.ab05od
.. autofunction:: slicot.ab05pd
.. autofunction:: slicot.ab05qd
.. autofunction:: slicot.ab05rd
.. autofunction:: slicot.ab05sd
.. autofunction:: slicot.ab07md
.. autofunction:: slicot.ab07nd
.. autofunction:: slicot.ab08md
.. autofunction:: slicot.ab08mz
.. autofunction:: slicot.ab08nd
.. autofunction:: slicot.ab08nw
.. autofunction:: slicot.ab08nx
.. autofunction:: slicot.ab08ny
.. autofunction:: slicot.ab08nz
.. autofunction:: slicot.ab09ad
.. autofunction:: slicot.ab09ax
.. autofunction:: slicot.ab09bd
.. autofunction:: slicot.ab09bx
.. autofunction:: slicot.ab09cd
.. autofunction:: slicot.ab09cx
.. autofunction:: slicot.ab09dd
.. autofunction:: slicot.ab09ed
.. autofunction:: slicot.ab09fd
.. autofunction:: slicot.ab09gd
.. autofunction:: slicot.ab09hd
.. autofunction:: slicot.ab09hx
.. autofunction:: slicot.ab09hy
.. autofunction:: slicot.ab09id
.. autofunction:: slicot.ab09ix
.. autofunction:: slicot.ab09iy
.. autofunction:: slicot.ab09jd
.. autofunction:: slicot.ab09jv
.. autofunction:: slicot.ab09jw
.. autofunction:: slicot.ab09jx
.. autofunction:: slicot.ab09kd
.. autofunction:: slicot.ab09kx
.. autofunction:: slicot.ab09md
.. autofunction:: slicot.ab09nd
.. autofunction:: slicot.ab13ad
.. autofunction:: slicot.ab13bd
.. autofunction:: slicot.ab13cd
.. autofunction:: slicot.ab13dd
.. autofunction:: slicot.ab13dx
.. autofunction:: slicot.ab13ed
.. autofunction:: slicot.ab13fd
.. autofunction:: slicot.ab13hd
.. autofunction:: slicot.ab13id
.. autofunction:: slicot.ab13md
.. autofunction:: slicot.ab8nxz

AG - Analysis - Generalized/Descriptor Systems
----------------------------------------------

Analysis routines for descriptor (generalized) state-space systems E*dx/dt = A*x + B*u,
including system inversion and transmission zeros.

.. autofunction:: slicot.ag07bd
.. autofunction:: slicot.ag08bd
.. autofunction:: slicot.ag08by
.. autofunction:: slicot.ag08bz
.. autofunction:: slicot.ag8byz

BB - Benchmark - Continuous-time Systems
----------------------------------------

Generate standard benchmark state-space models for testing control algorithms.

.. autofunction:: slicot.bb01ad
.. autofunction:: slicot.bb02ad
.. autofunction:: slicot.bb03ad
.. autofunction:: slicot.bb04ad

BD - Benchmark - Data Generation
--------------------------------

.. autofunction:: slicot.bd01ad
.. autofunction:: slicot.bd02ad

DE - Data Analysis - Exponential
--------------------------------

.. autofunction:: slicot.de01od
.. autofunction:: slicot.de01pd

DF - Data Analysis - FFT
------------------------

.. autofunction:: slicot.df01md

DG - Data Analysis - General
----------------------------

.. autofunction:: slicot.dg01md
.. autofunction:: slicot.dg01nd
.. autofunction:: slicot.dg01od
.. autofunction:: slicot.dgegv

DK - Data Analysis - Kalman
---------------------------

.. autofunction:: slicot.dk01md

DL - Data Analysis - Linear
---------------------------

.. autofunction:: slicot.dlatzm

FB - Factorization - Block
--------------------------

Kalman filtering: square-root and conventional Kalman filter implementations
for state estimation in linear dynamical systems.

.. autofunction:: slicot.fb01qd
.. autofunction:: slicot.fb01rd
.. autofunction:: slicot.fb01sd
.. autofunction:: slicot.fb01td
.. autofunction:: slicot.fb01vd

FD - Factorization - Descriptor
-------------------------------

.. autofunction:: slicot.fd01ad

IB - Identification - Block
---------------------------

Subspace-based system identification: estimate state-space models (A, B, C, D)
from measured input/output data. Includes MOESP, N4SID, and nonlinear (Wiener) methods.

.. autofunction:: slicot.ib01ad
.. autofunction:: slicot.ib01bd
.. autofunction:: slicot.ib01cd
.. autofunction:: slicot.ib01md
.. autofunction:: slicot.ib01nd
.. autofunction:: slicot.ib01od
.. autofunction:: slicot.ib01oy
.. autofunction:: slicot.ib01pd
.. autofunction:: slicot.ib01qd
.. autofunction:: slicot.ib01rd
.. autofunction:: slicot.ib03ad
.. autofunction:: slicot.ib03bd

MA - Matrix Operations - General
--------------------------------

.. autofunction:: slicot.ma01ad
.. autofunction:: slicot.ma01bd
.. autofunction:: slicot.ma01bz
.. autofunction:: slicot.ma01cd
.. autofunction:: slicot.ma01dd
.. autofunction:: slicot.ma01dz
.. autofunction:: slicot.ma02ad
.. autofunction:: slicot.ma02az
.. autofunction:: slicot.ma02bd
.. autofunction:: slicot.ma02bz
.. autofunction:: slicot.ma02cd
.. autofunction:: slicot.ma02cz
.. autofunction:: slicot.ma02dd
.. autofunction:: slicot.ma02ed
.. autofunction:: slicot.ma02es
.. autofunction:: slicot.ma02ez
.. autofunction:: slicot.ma02gd
.. autofunction:: slicot.ma02gz
.. autofunction:: slicot.ma02hd
.. autofunction:: slicot.ma02hz
.. autofunction:: slicot.ma02iz
.. autofunction:: slicot.ma02jd
.. autofunction:: slicot.ma02jz
.. autofunction:: slicot.ma02md
.. autofunction:: slicot.ma02mz
.. autofunction:: slicot.ma02nz
.. autofunction:: slicot.ma02od
.. autofunction:: slicot.ma02oz
.. autofunction:: slicot.ma02pd
.. autofunction:: slicot.ma02pz
.. autofunction:: slicot.ma02rd
.. autofunction:: slicot.ma02sd

MB - Matrix Operations - Block/Band
-----------------------------------

Matrix computations: symmetric/triangular updates, Cholesky factorizations, Toeplitz solvers,
eigenvalue reordering, periodic Schur decompositions, and matrix exponentials.

.. autofunction:: slicot.mb01kd
.. autofunction:: slicot.mb01ld
.. autofunction:: slicot.mb01md
.. autofunction:: slicot.mb01nd
.. autofunction:: slicot.mb01oc
.. autofunction:: slicot.mb01od
.. autofunction:: slicot.mb01oe
.. autofunction:: slicot.mb01oh
.. autofunction:: slicot.mb01oo
.. autofunction:: slicot.mb01os
.. autofunction:: slicot.mb01ot
.. autofunction:: slicot.mb01pd
.. autofunction:: slicot.mb01qd
.. autofunction:: slicot.mb01rb
.. autofunction:: slicot.mb01rd
.. autofunction:: slicot.mb01rh
.. autofunction:: slicot.mb01rt
.. autofunction:: slicot.mb01ru
.. autofunction:: slicot.mb01rw
.. autofunction:: slicot.mb01rx
.. autofunction:: slicot.mb01ry
.. autofunction:: slicot.mb01sd
.. autofunction:: slicot.mb01ss
.. autofunction:: slicot.mb01td
.. autofunction:: slicot.mb01ud
.. autofunction:: slicot.mb01uw
.. autofunction:: slicot.mb01ux
.. autofunction:: slicot.mb01uy
.. autofunction:: slicot.mb01uz
.. autofunction:: slicot.mb01wd
.. autofunction:: slicot.mb01xd
.. autofunction:: slicot.mb01xy
.. autofunction:: slicot.mb01yd
.. autofunction:: slicot.mb01zd
.. autofunction:: slicot.mb02cd
.. autofunction:: slicot.mb02cu
.. autofunction:: slicot.mb02cv
.. autofunction:: slicot.mb02cx
.. autofunction:: slicot.mb02cy
.. autofunction:: slicot.mb02dd
.. autofunction:: slicot.mb02ed
.. autofunction:: slicot.mb02fd
.. autofunction:: slicot.mb02gd
.. autofunction:: slicot.mb02hd
.. autofunction:: slicot.mb02id
.. autofunction:: slicot.mb02jd
.. autofunction:: slicot.mb02jx
.. autofunction:: slicot.mb02kd
.. autofunction:: slicot.mb02md
.. autofunction:: slicot.mb02nd
.. autofunction:: slicot.mb02ny
.. autofunction:: slicot.mb02od
.. autofunction:: slicot.mb02pd
.. autofunction:: slicot.mb02qd
.. autofunction:: slicot.mb02rd
.. autofunction:: slicot.mb02rz
.. autofunction:: slicot.mb02sd
.. autofunction:: slicot.mb02sz
.. autofunction:: slicot.mb02td
.. autofunction:: slicot.mb02tz
.. autofunction:: slicot.mb02ud
.. autofunction:: slicot.mb02uu
.. autofunction:: slicot.mb02uv
.. autofunction:: slicot.mb02uw
.. autofunction:: slicot.mb02vd
.. autofunction:: slicot.mb02wd
.. autofunction:: slicot.mb02yd
.. autofunction:: slicot.mb03ab
.. autofunction:: slicot.mb03ad
.. autofunction:: slicot.mb03ae
.. autofunction:: slicot.mb03ag
.. autofunction:: slicot.mb03ah
.. autofunction:: slicot.mb03ai
.. autofunction:: slicot.mb03ba
.. autofunction:: slicot.mb03bc
.. autofunction:: slicot.mb03bd
.. autofunction:: slicot.mb03be
.. autofunction:: slicot.mb03bf
.. autofunction:: slicot.mb03bg
.. autofunction:: slicot.mb03bz
.. autofunction:: slicot.mb03cd
.. autofunction:: slicot.mb03cz
.. autofunction:: slicot.mb03dd
.. autofunction:: slicot.mb03dz
.. autofunction:: slicot.mb03ed
.. autofunction:: slicot.mb03fd
.. autofunction:: slicot.mb03fz
.. autofunction:: slicot.mb03gd
.. autofunction:: slicot.mb03gz
.. autofunction:: slicot.mb03hd
.. autofunction:: slicot.mb03hz
.. autofunction:: slicot.mb03id
.. autofunction:: slicot.mb03iz
.. autofunction:: slicot.mb03jd
.. autofunction:: slicot.mb03jp
.. autofunction:: slicot.mb03jz
.. autofunction:: slicot.mb03ka
.. autofunction:: slicot.mb03kb
.. autofunction:: slicot.mb03kc
.. autofunction:: slicot.mb03kd
.. autofunction:: slicot.mb03ke
.. autofunction:: slicot.mb03ld
.. autofunction:: slicot.mb03lf
.. autofunction:: slicot.mb03lp
.. autofunction:: slicot.mb03lz
.. autofunction:: slicot.mb03md
.. autofunction:: slicot.mb03od
.. autofunction:: slicot.mb03oy
.. autofunction:: slicot.mb03pd
.. autofunction:: slicot.mb03py
.. autofunction:: slicot.mb03qd
.. autofunction:: slicot.mb03qg
.. autofunction:: slicot.mb03qv
.. autofunction:: slicot.mb03qw
.. autofunction:: slicot.mb03qx
.. autofunction:: slicot.mb03qy
.. autofunction:: slicot.mb03rd
.. autofunction:: slicot.mb03rw
.. autofunction:: slicot.mb03rx
.. autofunction:: slicot.mb03ry
.. autofunction:: slicot.mb03rz
.. autofunction:: slicot.mb03sd
.. autofunction:: slicot.mb03td
.. autofunction:: slicot.mb03ts
.. autofunction:: slicot.mb03ud
.. autofunction:: slicot.mb03vd
.. autofunction:: slicot.mb03vw
.. autofunction:: slicot.mb03vy
.. autofunction:: slicot.mb03wa
.. autofunction:: slicot.mb03wd
.. autofunction:: slicot.mb03wx
.. autofunction:: slicot.mb03xd
.. autofunction:: slicot.mb03xp
.. autofunction:: slicot.mb03xs
.. autofunction:: slicot.mb03xu
.. autofunction:: slicot.mb03xz
.. autofunction:: slicot.mb03ya
.. autofunction:: slicot.mb03yd
.. autofunction:: slicot.mb03yt
.. autofunction:: slicot.mb03za
.. autofunction:: slicot.mb03zd
.. autofunction:: slicot.mb04ad
.. autofunction:: slicot.mb04az
.. autofunction:: slicot.mb04bd
.. autofunction:: slicot.mb04bp
.. autofunction:: slicot.mb04bz
.. autofunction:: slicot.mb04cd
.. autofunction:: slicot.mb04db
.. autofunction:: slicot.mb04dd
.. autofunction:: slicot.mb04di
.. autofunction:: slicot.mb04dl
.. autofunction:: slicot.mb04dp
.. autofunction:: slicot.mb04ds
.. autofunction:: slicot.mb04dy
.. autofunction:: slicot.mb04dz
.. autofunction:: slicot.mb04ed
.. autofunction:: slicot.mb04fd
.. autofunction:: slicot.mb04fp
.. autofunction:: slicot.mb04gd
.. autofunction:: slicot.mb04hd
.. autofunction:: slicot.mb04id
.. autofunction:: slicot.mb04iy
.. autofunction:: slicot.mb04iz
.. autofunction:: slicot.mb04jd
.. autofunction:: slicot.mb04kd
.. autofunction:: slicot.mb04ld
.. autofunction:: slicot.mb04md
.. autofunction:: slicot.mb04nd
.. autofunction:: slicot.mb04ny
.. autofunction:: slicot.mb04od
.. autofunction:: slicot.mb04ow
.. autofunction:: slicot.mb04ox
.. autofunction:: slicot.mb04oy
.. autofunction:: slicot.mb04pa
.. autofunction:: slicot.mb04pb
.. autofunction:: slicot.mb04pu
.. autofunction:: slicot.mb04py
.. autofunction:: slicot.mb04qb
.. autofunction:: slicot.mb04qc
.. autofunction:: slicot.mb04qf
.. autofunction:: slicot.mb04qs
.. autofunction:: slicot.mb04qu
.. autofunction:: slicot.mb04rb
.. autofunction:: slicot.mb04rd
.. autofunction:: slicot.mb04rs
.. autofunction:: slicot.mb04rt
.. autofunction:: slicot.mb04ru
.. autofunction:: slicot.mb04rv
.. autofunction:: slicot.mb04rw
.. autofunction:: slicot.mb04rz
.. autofunction:: slicot.mb04su
.. autofunction:: slicot.mb04tb
.. autofunction:: slicot.mb04ts
.. autofunction:: slicot.mb04tt
.. autofunction:: slicot.mb04tu
.. autofunction:: slicot.mb04tv
.. autofunction:: slicot.mb04tw
.. autofunction:: slicot.mb04tx
.. autofunction:: slicot.mb04ty
.. autofunction:: slicot.mb04ud
.. autofunction:: slicot.mb04vd
.. autofunction:: slicot.mb04vx
.. autofunction:: slicot.mb04wd
.. autofunction:: slicot.mb04wp
.. autofunction:: slicot.mb04wr
.. autofunction:: slicot.mb04wu
.. autofunction:: slicot.mb04xd
.. autofunction:: slicot.mb04xy
.. autofunction:: slicot.mb04yd
.. autofunction:: slicot.mb04yw
.. autofunction:: slicot.mb04zd
.. autofunction:: slicot.mb05md
.. autofunction:: slicot.mb05my
.. autofunction:: slicot.mb05nd
.. autofunction:: slicot.mb05od
.. autofunction:: slicot.mb05oy
.. autofunction:: slicot.mb3jzp
.. autofunction:: slicot.mb3lzp
.. autofunction:: slicot.mb3oyz
.. autofunction:: slicot.mb3pyz
.. autofunction:: slicot.mb4dbz
.. autofunction:: slicot.mb4dlz
.. autofunction:: slicot.mb4dpz

MC - Matrix Operations - Complex
--------------------------------

Polynomial operations: evaluation, GCD, root finding, and polynomial matrix operations.

.. autofunction:: slicot.mc01md
.. autofunction:: slicot.mc01nd
.. autofunction:: slicot.mc01od
.. autofunction:: slicot.mc01pd
.. autofunction:: slicot.mc01py
.. autofunction:: slicot.mc01qd
.. autofunction:: slicot.mc01rd
.. autofunction:: slicot.mc01sd
.. autofunction:: slicot.mc01sw
.. autofunction:: slicot.mc01sx
.. autofunction:: slicot.mc01sy
.. autofunction:: slicot.mc01td
.. autofunction:: slicot.mc01vd
.. autofunction:: slicot.mc01wd
.. autofunction:: slicot.mc01xd
.. autofunction:: slicot.mc03md
.. autofunction:: slicot.mc03nd
.. autofunction:: slicot.mc03nx
.. autofunction:: slicot.mc03ny

MD - Matrix Operations - Descriptor
-----------------------------------

.. autofunction:: slicot.md03ba
.. autofunction:: slicot.md03bb
.. autofunction:: slicot.md03bd
.. autofunction:: slicot.md03bf
.. autofunction:: slicot.md03by

NF - Numerical Functions
------------------------

.. autofunction:: slicot.nf01ay
.. autofunction:: slicot.nf01br
.. autofunction:: slicot.nf01bs
.. autofunction:: slicot.nf01by

SB - Synthesis - Basic
----------------------

Controller design: Riccati/Lyapunov/Sylvester equation solvers, pole assignment,
H-infinity control, spectral factorization, and coprime factorization.

.. autofunction:: slicot.sb01bd
.. autofunction:: slicot.sb01bx
.. autofunction:: slicot.sb01by
.. autofunction:: slicot.sb01dd
.. autofunction:: slicot.sb01fy
.. autofunction:: slicot.sb01md
.. autofunction:: slicot.sb02cx
.. autofunction:: slicot.sb02md
.. autofunction:: slicot.sb02mr
.. autofunction:: slicot.sb02ms
.. autofunction:: slicot.sb02mt
.. autofunction:: slicot.sb02mu
.. autofunction:: slicot.sb02mv
.. autofunction:: slicot.sb02mw
.. autofunction:: slicot.sb02mx
.. autofunction:: slicot.sb02nd
.. autofunction:: slicot.sb02od
.. autofunction:: slicot.sb02ou
.. autofunction:: slicot.sb02ov
.. autofunction:: slicot.sb02ow
.. autofunction:: slicot.sb02ox
.. autofunction:: slicot.sb02oy
.. autofunction:: slicot.sb02pd
.. autofunction:: slicot.sb02qd
.. autofunction:: slicot.sb02rd
.. autofunction:: slicot.sb02ru
.. autofunction:: slicot.sb02sd
.. autofunction:: slicot.sb03md
.. autofunction:: slicot.sb03mu
.. autofunction:: slicot.sb03mv
.. autofunction:: slicot.sb03mw
.. autofunction:: slicot.sb03mx
.. autofunction:: slicot.sb03my
.. autofunction:: slicot.sb03od
.. autofunction:: slicot.sb03ou
.. autofunction:: slicot.sb03ov
.. autofunction:: slicot.sb03pd
.. autofunction:: slicot.sb03qd
.. autofunction:: slicot.sb03qx
.. autofunction:: slicot.sb03qy
.. autofunction:: slicot.sb03rd
.. autofunction:: slicot.sb03sd
.. autofunction:: slicot.sb03sx
.. autofunction:: slicot.sb03sy
.. autofunction:: slicot.sb03td
.. autofunction:: slicot.sb03ud
.. autofunction:: slicot.sb04md
.. autofunction:: slicot.sb04mr
.. autofunction:: slicot.sb04mw
.. autofunction:: slicot.sb04nd
.. autofunction:: slicot.sb04nv
.. autofunction:: slicot.sb04nw
.. autofunction:: slicot.sb04nx
.. autofunction:: slicot.sb04ny
.. autofunction:: slicot.sb04od
.. autofunction:: slicot.sb04ow
.. autofunction:: slicot.sb04pd
.. autofunction:: slicot.sb04py
.. autofunction:: slicot.sb04qd
.. autofunction:: slicot.sb04qr
.. autofunction:: slicot.sb04rd
.. autofunction:: slicot.sb04rv
.. autofunction:: slicot.sb04rw
.. autofunction:: slicot.sb04rx
.. autofunction:: slicot.sb04ry
.. autofunction:: slicot.sb06nd
.. autofunction:: slicot.sb08cd
.. autofunction:: slicot.sb08dd
.. autofunction:: slicot.sb08ed
.. autofunction:: slicot.sb08fd
.. autofunction:: slicot.sb08gd
.. autofunction:: slicot.sb08hd
.. autofunction:: slicot.sb08md
.. autofunction:: slicot.sb08my
.. autofunction:: slicot.sb08nd
.. autofunction:: slicot.sb08ny
.. autofunction:: slicot.sb09md
.. autofunction:: slicot.sb10ad
.. autofunction:: slicot.sb10dd
.. autofunction:: slicot.sb10ed
.. autofunction:: slicot.sb10fd
.. autofunction:: slicot.sb10hd
.. autofunction:: slicot.sb10id
.. autofunction:: slicot.sb10jd
.. autofunction:: slicot.sb10kd
.. autofunction:: slicot.sb10ld
.. autofunction:: slicot.sb10md
.. autofunction:: slicot.sb10pd
.. autofunction:: slicot.sb10rd
.. autofunction:: slicot.sb10sd
.. autofunction:: slicot.sb10td
.. autofunction:: slicot.sb10vd
.. autofunction:: slicot.sb10wd
.. autofunction:: slicot.sb10yd
.. autofunction:: slicot.sb10zd
.. autofunction:: slicot.sb10zp
.. autofunction:: slicot.sb16ad
.. autofunction:: slicot.sb16bd
.. autofunction:: slicot.sb16cd
.. autofunction:: slicot.sb16cy

SG - Synthesis - Generalized/Descriptor
---------------------------------------

Generalized Riccati and Lyapunov equation solvers for descriptor systems.

.. autofunction:: slicot.sg02ad
.. autofunction:: slicot.sg02cv
.. autofunction:: slicot.sg02cw
.. autofunction:: slicot.sg02cx
.. autofunction:: slicot.sg02nd
.. autofunction:: slicot.sg03ad
.. autofunction:: slicot.sg03ax
.. autofunction:: slicot.sg03ay
.. autofunction:: slicot.sg03bd
.. autofunction:: slicot.sg03br
.. autofunction:: slicot.sg03bs
.. autofunction:: slicot.sg03bt
.. autofunction:: slicot.sg03bu
.. autofunction:: slicot.sg03bv
.. autofunction:: slicot.sg03bw
.. autofunction:: slicot.sg03bx
.. autofunction:: slicot.sg03by
.. autofunction:: slicot.sg03bz

TB - Transformation - Basic
---------------------------

State-space transformations: minimal realization, balancing, similarity transforms,
and conversion between state-space and transfer function representations.

.. autofunction:: slicot.tb01id
.. autofunction:: slicot.tb01iz
.. autofunction:: slicot.tb01kd
.. autofunction:: slicot.tb01kx
.. autofunction:: slicot.tb01ld
.. autofunction:: slicot.tb01md
.. autofunction:: slicot.tb01nd
.. autofunction:: slicot.tb01pd
.. autofunction:: slicot.tb01px
.. autofunction:: slicot.tb01td
.. autofunction:: slicot.tb01ty
.. autofunction:: slicot.tb01ud
.. autofunction:: slicot.tb01ux
.. autofunction:: slicot.tb01uy
.. autofunction:: slicot.tb01vd
.. autofunction:: slicot.tb01vy
.. autofunction:: slicot.tb01wd
.. autofunction:: slicot.tb01wx
.. autofunction:: slicot.tb01xd
.. autofunction:: slicot.tb01xz
.. autofunction:: slicot.tb01zd
.. autofunction:: slicot.tb03ad
.. autofunction:: slicot.tb03ay
.. autofunction:: slicot.tb04ad
.. autofunction:: slicot.tb04bd
.. autofunction:: slicot.tb04bv
.. autofunction:: slicot.tb04bw
.. autofunction:: slicot.tb04bx
.. autofunction:: slicot.tb04cd
.. autofunction:: slicot.tb05ad

TC - Transformation - Complex
-----------------------------

.. autofunction:: slicot.tc01od
.. autofunction:: slicot.tc04ad
.. autofunction:: slicot.tc05ad

TD - Transformation - Descriptor
--------------------------------

.. autofunction:: slicot.td03ad
.. autofunction:: slicot.td03ay
.. autofunction:: slicot.td04ad
.. autofunction:: slicot.td05ad

TF - Transfer Function
----------------------

Transfer function evaluation: compute frequency response and time-domain output
of state-space systems.

.. autofunction:: slicot.tf01md
.. autofunction:: slicot.tf01mx
.. autofunction:: slicot.tf01my
.. autofunction:: slicot.tf01nd
.. autofunction:: slicot.tf01od
.. autofunction:: slicot.tf01pd
.. autofunction:: slicot.tf01qd
.. autofunction:: slicot.tf01rd

TG - Transformation - Generalized
---------------------------------

Transformations for descriptor systems: reduction to SVD-like, Hessenberg-triangular,
and block diagonal forms.

.. autofunction:: slicot.tg01ad
.. autofunction:: slicot.tg01az
.. autofunction:: slicot.tg01bd
.. autofunction:: slicot.tg01cd
.. autofunction:: slicot.tg01dd
.. autofunction:: slicot.tg01ed
.. autofunction:: slicot.tg01fd
.. autofunction:: slicot.tg01fz
.. autofunction:: slicot.tg01gd
.. autofunction:: slicot.tg01hd
.. autofunction:: slicot.tg01hu
.. autofunction:: slicot.tg01hx
.. autofunction:: slicot.tg01hy
.. autofunction:: slicot.tg01id
.. autofunction:: slicot.tg01jd
.. autofunction:: slicot.tg01jy
.. autofunction:: slicot.tg01kd
.. autofunction:: slicot.tg01kz
.. autofunction:: slicot.tg01ld
.. autofunction:: slicot.tg01ly
.. autofunction:: slicot.tg01md
.. autofunction:: slicot.tg01nd
.. autofunction:: slicot.tg01nx
.. autofunction:: slicot.tg01oa
.. autofunction:: slicot.tg01ob
.. autofunction:: slicot.tg01od
.. autofunction:: slicot.tg01oz
.. autofunction:: slicot.tg01pd
.. autofunction:: slicot.tg01qd
.. autofunction:: slicot.tg01wd

UD - Utility - Descriptor
-------------------------

.. autofunction:: slicot.ud01bd
.. autofunction:: slicot.ud01cd
.. autofunction:: slicot.ud01dd
.. autofunction:: slicot.ud01md
.. autofunction:: slicot.ud01mz
.. autofunction:: slicot.ud01nd

UE - Utility - Elementary
-------------------------

.. autofunction:: slicot.ue01md

ZG - Zeros - Generalized
------------------------

.. autofunction:: slicot.zgegs
.. autofunction:: slicot.zgegv


Complete Function List
----------------------

.. code-block:: python

   import slicot
   print([name for name in dir(slicot) if not name.startswith('_')])

