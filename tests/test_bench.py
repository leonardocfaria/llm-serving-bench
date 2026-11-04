import bench


def test_quick_benchmark_writes_table_and_plot(tmp_path):
    res = bench.main(quick=True, out=str(tmp_path), max_new=4)
    assert set(res["load"]) == set(bench.CONFIGS)
    for rows in res["load"].values():
        assert all(r["p95_ms"] >= r["p50_ms"] > 0 and r["tokens_per_s"] > 0 for r in rows)
    assert all(r["mean_batch"] == 1 for r in res["load"]["naive-fp32"])
    assert res["quant"]["int8_mb"] < res["quant"]["fp32_mb"]
    assert "| batched-int8 |" in (tmp_path / "results.md").read_text()
    assert (tmp_path / "bench.png").stat().st_size > 0
