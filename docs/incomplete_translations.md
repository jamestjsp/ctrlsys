# Current SLICOT Translation Gap Audit

Refreshed 2026-05-10 against the current C sources and
`SLICOT-Reference/src`. The older 2025 line-ratio table is no longer reliable:
several files it called stubs now contain substantial translated logic.

## Summary

| Classification | Count | Meaning |
| --- | ---: | --- |
| Active parity task | 0 | No current behavior gap remains open in the refreshed audit. |
| Needs differential proof | 0 | The risky paths identified by this audit now have focused proof or a task result documenting why no fix was needed. |
| Completed / stale entry | 13 | The missing calls, stub claims, or risky paths from the old audit are now closed by implementation, differential coverage, or branch-specific proof. |
| Deliberate replacement / inlining | 5 | The C code intentionally uses equivalent wrappers, expanded arithmetic, or inline loops. |
| Performance-only degradation | 1 | Behavior is expected to be correct; remaining gap is optimization. |

## Closed parity tasks

| Routine | Current status | Tracking |
| --- | --- | --- |
| `ab13dd.c` | The continuous-time candidate-frequency ordering path has branch coverage and Fortran-reference proof. | `LZ6EVQ` |
| `mb03bd.c` | The `DLAS2` singularity-test concern is covered by focused parity proof. | `KJDQQC` |
| `mb01uy.c` | Workspace-query semantics now match the Fortran `DGEQRF` query contract. | `XDMHPE` |
| `mb02cu.c` | Blocked QR/LQ paths and blocked `B` reflector-scalar side effects now have parity evidence. | `2ZMHMW`, `VERVRD` |

## Differential proof added

| Routine | Current finding |
| --- | --- |
| `ab13hd.c` | The old "4% identity-only" claim is stale; current C contains the broader descriptor/generalized code paths and `DLASRT` calls, with targeted coverage and Fortran differential tests for risky branches. |
| `ab09iy.c` | The old missing `MB01WD`/`DSYEV` claim is stale; current C calls both and now has focused weighted-Gramian differential coverage. |
| `mb01vd.c` | Sparse/dense Kronecker product branches now have branch-matrix differential coverage across transpose combinations. |
| `mb04rb.c` | The blocked symplectic update concern has been audited and covered by a narrow parity task. |
| `mb04tb.c` | The blocked path has Fortran differential coverage; the old stub/delegation claim is stale. |

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
