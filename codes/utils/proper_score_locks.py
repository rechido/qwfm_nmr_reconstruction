"""Independent model writers and short publication locks for proper scores."""

from contextlib import contextmanager
import fcntl
from pathlib import Path


@contextmanager
def file_lock(path: Path, *, shared: bool = False, blocking: bool = False,
              message: str = "Another process is writing this artifact."):
    path.parent.mkdir(parents=True, exist_ok=True)
    # NFS implements shared flock locks as read locks, so the descriptor must
    # allow reading as well as writing. Append mode preserves existing files.
    with path.open("a+") as handle:
        flags = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        if not blocking:
            flags |= fcntl.LOCK_NB
        try:
            fcntl.flock(handle, flags)
        except BlockingIOError as error:
            raise RuntimeError(message) from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def model_generation_lock(run_dir: Path, model_name: str):
    if not model_name or Path(model_name).name != model_name or model_name in (".", ".."):
        raise ValueError("Model lock requires one model directory name.")
    # New jobs share this compatibility lock, so different models never wait on
    # one another. Old code held it exclusively for the entire experiment.
    with file_lock(run_dir / ".generation.lock", shared=True, message=(
        "An older generator or test-condition preparation is holding the run lock. "
        "Finish that job, then use the updated model scripts; do not delete the lock file."
    )):
        with file_lock(run_dir / ".model_locks" / f"{model_name}.lock", message=(
            f"Another process is generating the same model: {model_name}. "
            "Different models can run concurrently."
        )):
            yield
