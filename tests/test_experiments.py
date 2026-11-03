from dico_nli.experiments import create_run


def test_new_runs_do_not_overwrite_previous_artifacts(tmp_path):
    first = create_run(tmp_path)
    sentinel = first / "completed_artifact.txt"
    sentinel.write_text("preserve")
    second = create_run(tmp_path)
    assert second != first
    assert sentinel.read_text() == "preserve"
