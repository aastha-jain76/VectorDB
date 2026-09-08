#!/usr/bin/env bash
set -e

echo "=== Running Vector Database Benchmark Suite ==="
python3 evaluation/benchmark.py
python3 evaluation/reporter.py

echo ""
echo "=== Benchmark Complete ==="
echo "Report generated at: docs/benchmark_report.md"
echo "Trade-off curve saved at: evaluation/results/tradeoff_curve.png"
