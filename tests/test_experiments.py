import logging
import sys
from contextlib import redirect_stderr
from io import StringIO

from dico_nli.experiments import _Tee, create_run, training_log


def test_new_runs_do_not_overwrite_previous_artifacts(tmp_path):
    first = create_run(tmp_path)
    sentinel = first / "completed_artifact.txt"
    sentinel.write_text("preserve")
    second = create_run(tmp_path)
    assert second != first
    assert sentinel.read_text() == "preserve"


def test_retained_logger_stream_survives_training_log_shutdown(tmp_path):
    console = StringIO()
    logger = logging.Logger("retained-training-stream", level=logging.INFO)
    handler = None
    try:
        with redirect_stderr(console):
            with training_log(tmp_path):
                # Libraries can retain the redirected stream until interpreter exit.
                handler = logging.StreamHandler(sys.stderr)
                logger.addHandler(handler)
                logger.info("inside training")
            # absl's handler closes its retained stream during logging.shutdown().
            handler.stream.close()
            handler.stream.close()
            handler.flush()
            logger.info("after training")
        assert not console.closed
        assert console.getvalue() == "inside training\nafter training\n"
        assert (tmp_path / "training.log").read_text() == "inside training\n"
    finally:
        if handler is not None:
            logger.removeHandler(handler)
            handler.stream = console
            handler.close()


def test_tee_close_does_not_own_either_stream_and_write_returns_count():
    console, log = StringIO(), StringIO()
    stream = _Tee(console, log)
    stream.close()
    assert not console.closed and not log.closed
    assert stream.write("still open") == len("still open")
    assert stream.isatty() == console.isatty()
    assert stream.encoding == console.encoding
    assert console.getvalue() == log.getvalue() == "still open"
