import json

from aspectbench.runtime.runs import RunLayout, record_id


def test_record_id_is_stable_and_hides_text():
    row = {"article": "A private-looking sentence", "aspect": "Example"}
    assert record_id(row) == record_id(dict(row))
    assert "private" not in record_id(row).lower()


def test_run_layout_collects_and_resumes(tmp_path):
    layout = RunLayout("infer", "test", run_root=tmp_path)
    layout.write_shard("xlmr-hbs-masked", 0, [{"record_id": "one", "prediction": 1}])
    layout.write_shard("xlmr-hbs-masked", 1, [{"record_id": "two", "prediction": -1}])
    assert layout.completed_ids("xlmr-hbs-masked") == {"one", "two"}
    assert [row["record_id"] for row in layout.collect("xlmr-hbs-masked")] == ["one", "two"]
    layout.update_progress(status="running", completed=1)
    layout.update_progress(status="complete")
    progress = json.loads(layout.progress_path.read_text())
    assert progress["completed"] == 1
    assert progress["status"] == "complete"


def test_no_resume_rejects_existing_run(tmp_path):
    RunLayout("infer", "test", run_root=tmp_path)
    try:
        RunLayout("infer", "test", run_root=tmp_path, resume=False)
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing run should be rejected without resume")
