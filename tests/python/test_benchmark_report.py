import importlib.util
from pathlib import Path


def load_benchmark_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_c_vs_fortran.py"
    spec = importlib.util.spec_from_file_location("benchmark_c_vs_fortran", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_report_lists_fortran_faster_rows_sorted_by_advantage():
    bench = load_benchmark_module()
    comparisons = [
        bench.ComparisonResult("bb04ad", "ex4.1", 50, 40.0, 20.0, 0.5, 0.1, 0.1, 0, 0),
        bench.ComparisonResult("bb03ad", "ex4.2", 30, 12.0, 10.0, 0.833333, 0.1, 0.1, 0, 0),
        bench.ComparisonResult("bb04ad", "ex4.3", 20, 2.0, 4.0, 2.0, 0.1, 0.1, 0, 0),
    ]

    report = bench.generate_markdown_report([], [], comparisons, {})

    assert "## Fortran Faster Rows" in report
    assert "| Routine | Dataset | N | C11 (μs) | F77 (μs) | F77/C11 Ratio | Delta (μs) |" in report
    first = report.index("| BB04AD | ex4.1 | 50 | 40.00 | 20.00 | 2.00x | 20.00 |")
    second = report.index("| BB03AD | ex4.2 | 30 | 12.00 | 10.00 | 1.20x | 2.00 |")
    assert first < second
    assert "| BB04AD | ex4.3 | 20 |" not in report


def test_report_states_when_no_fortran_rows_are_faster():
    bench = load_benchmark_module()
    comparisons = [
        bench.ComparisonResult("bb04ad", "ex4.1", 50, 8.0, 20.0, 2.5, 0.1, 0.1, 0, 0),
    ]

    report = bench.generate_markdown_report([], [], comparisons, {})

    assert "## Fortran Faster Rows" in report
    assert "*No rows where Fortran is faster.*" in report
