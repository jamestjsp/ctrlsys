# Current SLICOT Translation Gap Audit

Refreshed 2026-05-10 against the current C sources and
`SLICOT-Reference/src`. The older 2025 line-ratio table is no longer reliable:
several files it called stubs now contain substantial translated logic.

## Summary

| Classification | Count | Meaning |
| --- | ---: | --- |
| Active parity task | 4 | Current evidence points to a specific behavior contract to prove or fix. |
| Needs differential proof | 5 | Current implementation is substantial, but the old audit identified a risky path that is not yet proved here. |
| Completed / stale entry | 4 | The missing calls or stub claims from the old audit are now present in C. |
| Deliberate replacement / inlining | 5 | The C code intentionally uses equivalent wrappers, expanded arithmetic, or inline loops. |
| Performance-only degradation | 1 | Behavior is expected to be correct; remaining gap is optimization. |

## Active parity tasks

| Routine | Current status | Tracking |
| --- | --- | --- |
| `ab13dd.c` | Still lacks an obvious `DLASRT` equivalent for continuous-time candidate frequencies. | `LZ6EVQ` |
| `mb03bd.c` | Current C no longer has the old zero-chasing TODO, but the Fortran `DLAS2` singularity test still needs parity proof. | `KJDQQC` |
| `mb01uy.c` | Element-wise update paths are present; workspace-query semantics still differ from the Fortran `DGEQRF` query. | `XDMHPE` |
| `mb02cu.c` | A2 blocked QR/LQ paths now have Fortran differential coverage; blocked `B` reflector-scalar side effects need deeper proof. | `2ZMHMW`, `VERVRD` |

## Needs differential proof

| Routine | Current finding |
| --- | --- |
| `ab13hd.c` | The old "4% identity-only" claim is stale; current C contains the broader descriptor/generalized code paths and `DLASRT` calls. Keep relying on targeted coverage and Fortran differential tests for risky branches rather than the old ratio. |
| `ab09iy.c` | The old missing `MB01WD`/`DSYEV` claim is stale; current C calls both for the alpha reweighting paths. Needs a focused weighted-Gramian differential before closing fully. |
| `mb01vd.c` | Current C implements the sparse/dense Kronecker product branches with `DCOPY`/scaling helpers. Needs branch-matrix differential coverage across transpose combinations before closing. |
| `mb04rb.c` | Current C still carries only a partial blocked symplectic URV update; the old `MB04PA` gap remains suspicious. Needs a narrow MB04RB task if this routine becomes priority. |
| `mb04tb.c` | Current C contains blocked `MB03XU`/`DGEMM` logic and delegates only the tail to `mb04ts`; the old stub/delegation claim is stale, but blocked-path parity still needs proof. |

## Completed or stale entries

| Routine | Current finding |
| --- | --- |
| `mb02pd.c` | Current C calls `DGEEQU`, `DLAQGE`, and `DGERFS`; the old equilibration/refinement gap is stale. |
| `mb04pb.c` | Current C calls `DGEHRD`, `MB04PA`, `DGEMM`, `DSYR2K`, and `MB04PU`; the old validation-only stub claim is stale. |
| `ab09iy.c` | No longer missing `MB01WD` or `DSYEV`; retained above only because differential proof is still useful. |
| `mb04tb.c` | No longer delegates entirely to `mb04ts`; retained above only because blocked-path proof is still useful. |

## Deliberate replacement / inlining

| Routine | Current finding |
| --- | --- |
| `mb03bb.c` | `DLADIV`, `ZLARTG`, and `ZROT` are expanded inline as complex arithmetic. |
| `mc01td.c` | `DRSCL` is replaced with explicit reciprocal scaling. |
| `mb03bf.c` | `DROT` is implemented inline. |
| `sg03bd.c` | Uses modern `DGGES` instead of deprecated `DGEGS`. |
| `tg01jy.c` | Uses the current helper path instead of the older Fortran helper name. |

## Performance-only degradation

| Routine | Current finding |
| --- | --- |
| `mb03xp.c` | Still delegates to the single-shift path instead of the Fortran multishift bulge-chasing optimization. Treat this as performance work unless a differential test proves a behavior gap. |

## Current gates

- `scripts/check_translation_smells.py`
- `scripts/check_translation_coverage.py --report-only`
- `python3 -m py_compile scripts/check_translation_smells.py scripts/check_translation_coverage.py`

Unresolved questions: none
