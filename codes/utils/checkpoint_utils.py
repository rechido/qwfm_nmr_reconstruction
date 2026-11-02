from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import uuid
import warnings
from pathlib import Path
from typing import Any, Mapping

import torch
from accelerate.utils import broadcast_object_list

try:
    import fcntl
except ImportError:  # pragma: no cover - the experiment targets Linux servers.
    fcntl = None


CHECKPOINT_SCHEMA_VERSION = 4
SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS = frozenset({1, 2, 3, 4})
MANIFEST_NAME = "checkpoint_manifest.json"
SUCCESS_NAME = "_SUCCESS"
_DATALOADER_GENERATOR_ROLE = "dataloader_shuffle_generator"
_EMA_ROLE = "ema"

_MODEL_STATE_PATTERN = re.compile(
    r"^(?:model|pytorch_model)(?:_\d+)?\.(?:safetensors|bin)$"
)
_OPTIMIZER_STATE_PATTERN = re.compile(r"^optimizer(?:_\d+)?\.bin$")
_SCHEDULER_STATE_PATTERN = re.compile(r"^scheduler(?:_\d+)?\.bin$")
_RNG_STATE_PATTERN = re.compile(r"^random_states_\d+\.pkl$")
_CUSTOM_STATE_PATTERN = re.compile(r"^custom_checkpoint_\d+\.pkl$")


class CheckpointValidationError(RuntimeError):
    """Raised when a checkpoint exists but is not a complete committed state."""


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, os.O_RDONLY | directory_flag)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        # Some network filesystems do not expose directory fsync. The atomic
        # same-filesystem rename remains the namespace-commit boundary.
        pass
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointValidationError(f"Cannot read checkpoint JSON {path}.") from error
    if not isinstance(payload, dict):
        raise CheckpointValidationError(f"Expected a JSON object in {path}.")
    return payload


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _build_inventory(directory: Path) -> list[dict[str, Any]]:
    inventory = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise CheckpointValidationError(
                f"Checkpoint generations must not contain symlinks: {path}"
            )
        if not path.is_file():
            continue
        relative = path.relative_to(directory).as_posix()
        if relative in {MANIFEST_NAME, SUCCESS_NAME}:
            continue
        inventory.append({"path": relative, "size": int(path.stat().st_size)})
    return inventory


def _classify_accelerate_state(
    inventory: list[dict[str, Any]],
) -> dict[str, list[str]]:
    roles = {
        "model": [],
        "optimizer": [],
        "scheduler": [],
        "scaler": [],
        "rng": [],
        "custom": [],
    }
    for record in inventory:
        relative = str(record["path"])
        if "/" in relative:
            continue
        name = Path(relative).name
        if _MODEL_STATE_PATTERN.fullmatch(name):
            roles["model"].append(relative)
        elif _OPTIMIZER_STATE_PATTERN.fullmatch(name):
            roles["optimizer"].append(relative)
        elif _SCHEDULER_STATE_PATTERN.fullmatch(name):
            roles["scheduler"].append(relative)
        elif name == "scaler.pt":
            roles["scaler"].append(relative)
        elif _RNG_STATE_PATTERN.fullmatch(name):
            roles["rng"].append(relative)
        elif _CUSTOM_STATE_PATTERN.fullmatch(name):
            roles["custom"].append(relative)
    for names in roles.values():
        names.sort()
    return roles


def _has_diffusers_weights(directory: Path) -> bool:
    if not (directory / "config.json").is_file():
        return False
    return any(
        path.is_file() and path.suffix in {".safetensors", ".bin"}
        for path in directory.iterdir()
    )


def _validate_state_roles(
    directory: Path,
    roles: Mapping[str, list[str]],
    *,
    world_size: int,
    has_scaler: bool,
    has_ema: bool,
    custom_checkpoint_roles: list[str] | None = None,
) -> None:
    for required_role in ("model", "optimizer", "scheduler"):
        if not roles.get(required_role):
            raise CheckpointValidationError(
                f"Accelerate {required_role} state is missing from {directory}."
            )
    rng_files = roles.get("rng", [])
    expected_rng_files = {
        f"random_states_{process_index}.pkl"
        for process_index in range(world_size)
    }
    if set(rng_files) != expected_rng_files:
        raise CheckpointValidationError(
            f"Expected per-rank RNG files {sorted(expected_rng_files)} in "
            f"{directory}, found {sorted(rng_files)}."
        )
    scaler_files = roles.get("scaler", [])
    if has_scaler and not scaler_files:
        raise CheckpointValidationError(
            f"The active GradScaler state is missing from {directory}."
        )
    if not has_scaler and scaler_files:
        raise CheckpointValidationError(
            f"Unexpected GradScaler state was found in {directory}."
        )
    custom_files = roles.get("custom", [])
    if custom_checkpoint_roles is not None:
        expected_custom_files = [
            f"custom_checkpoint_{index}.pkl"
            for index in range(len(custom_checkpoint_roles))
        ]
        if custom_files != expected_custom_files:
            raise CheckpointValidationError(
                "Accelerate custom-state files do not match their declared "
                f"registration order in {directory}: expected "
                f"{expected_custom_files} for roles {custom_checkpoint_roles}, "
                f"found {custom_files}."
            )
    elif has_ema and not custom_files:
        raise CheckpointValidationError(
            f"Registered EMA custom state is missing from {directory}."
        )
    if not _has_diffusers_weights(directory / "unet"):
        raise CheckpointValidationError(
            f"Raw Diffusers evaluation weights are incomplete in {directory / 'unet'}."
        )
    if has_ema and not _has_diffusers_weights(directory / "unet_ema"):
        raise CheckpointValidationError(
            f"EMA Diffusers evaluation weights are incomplete in "
            f"{directory / 'unet_ema'}."
        )


def _validate_checkpoint_schema_version(value: Any, *, location: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value not in SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS
    ):
        raise CheckpointValidationError(
            f"Unsupported {location} schema: {value!r}."
        )
    return value


def _expected_custom_checkpoint_roles(
    *,
    has_ema: bool,
    schema_version: int,
) -> list[str]:
    # Schema v3 checkpointed the finite shuffled DataLoader's generator.  The
    # iteration-native stream introduced in schema v4 is stateless: its next
    # batch is reconstructed solely from ``global_iteration``.
    roles = [] if schema_version >= 4 else [_DATALOADER_GENERATOR_ROLE]
    if has_ema:
        roles.append(_EMA_ROLE)
    return roles


def _validate_custom_checkpoint_roles(
    payload: Mapping[str, Any],
    schema_version: int,
    *,
    has_ema: bool,
    location: str,
) -> list[str] | None:
    """Validate the ordered Accelerate custom-state contract for schema v3+."""

    if schema_version < 3:
        if "custom_checkpoint_roles" in payload:
            raise CheckpointValidationError(
                f"{location} schema v{schema_version} must not declare "
                "custom_checkpoint_roles."
            )
        return None
    raw_roles = payload.get("custom_checkpoint_roles")
    expected_roles = _expected_custom_checkpoint_roles(
        has_ema=has_ema,
        schema_version=schema_version,
    )
    if raw_roles != expected_roles:
        raise CheckpointValidationError(
            f"{location} custom_checkpoint_roles must be {expected_roles}, "
            f"found {raw_roles!r}."
        )
    return expected_roles


def _validate_checkpoint_iteration(
    payload: Mapping[str, Any],
    schema_version: int,
    *,
    location: str,
) -> int:
    field = "global_step" if schema_version == 1 else "global_iteration"
    incompatible_field = (
        "global_iteration" if schema_version == 1 else "global_step"
    )
    if incompatible_field in payload:
        raise CheckpointValidationError(
            f"{location} schema v{schema_version} must not contain "
            f"{incompatible_field}."
        )
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CheckpointValidationError(
            f"{location} {field} must be a nonnegative integer."
        )
    return value


def _validate_optional_completed_epochs(
    payload: Mapping[str, Any],
    *,
    schema_version: int,
    location: str,
) -> int | None:
    """Validate the legacy epoch counter and reject it in iteration schema v4."""

    if "completed_epochs" not in payload:
        return None
    if schema_version >= 4:
        raise CheckpointValidationError(
            f"{location} schema v{schema_version} must not contain completed_epochs."
        )
    value = payload["completed_epochs"]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CheckpointValidationError(
            f"{location} completed_epochs must be a nonnegative integer."
        )
    return value


def _validate_inventory(
    directory: Path,
    manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    raw_inventory = manifest.get("inventory")
    if not isinstance(raw_inventory, list):
        raise CheckpointValidationError("Checkpoint manifest lacks an inventory list.")
    inventory: list[dict[str, Any]] = []
    expected_paths: set[str] = set()
    for raw_record in raw_inventory:
        if not isinstance(raw_record, dict):
            raise CheckpointValidationError("Invalid checkpoint inventory record.")
        relative = raw_record.get("path")
        size = raw_record.get("size")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or relative in expected_paths
        ):
            raise CheckpointValidationError(
                f"Unsafe or duplicate checkpoint inventory path: {relative!r}."
            )
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise CheckpointValidationError(
                f"Invalid checkpoint inventory size for {relative!r}."
            )
        path = directory / relative
        if path.is_symlink() or not path.is_file():
            raise CheckpointValidationError(
                f"Checkpoint inventory file is missing: {path}."
            )
        actual_size = int(path.stat().st_size)
        if actual_size != size:
            raise CheckpointValidationError(
                f"Checkpoint file size mismatch for {path}: "
                f"manifest={size}, actual={actual_size}."
            )
        expected_paths.add(relative)
        inventory.append({"path": relative, "size": size})

    actual_paths = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name not in {MANIFEST_NAME, SUCCESS_NAME}
    }
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        unexpected = sorted(actual_paths - expected_paths)
        raise CheckpointValidationError(
            f"Checkpoint inventory mismatch; missing={missing}, unexpected={unexpected}."
        )
    roles = _classify_accelerate_state(inventory)
    manifest_roles = manifest.get("accelerate_state_files")
    if manifest_roles != roles:
        raise CheckpointValidationError(
            "Checkpoint Accelerate-state role inventory does not match its files."
        )
    return inventory, roles


def _validate_committed_generation(
    directory: Path,
    *,
    require_ema: bool,
) -> tuple[int, dict[str, Any]]:
    manifest_path = directory / MANIFEST_NAME
    success_path = directory / SUCCESS_NAME
    if not manifest_path.is_file() or not success_path.is_file():
        raise CheckpointValidationError(
            f"Checkpoint generation is not committed: {directory}."
        )
    try:
        manifest_bytes = manifest_path.read_bytes()
        success_digest = success_path.read_text(encoding="ascii").strip()
    except OSError as error:
        raise CheckpointValidationError(
            f"Cannot read checkpoint commit metadata in {directory}."
        ) from error
    actual_digest = hashlib.sha256(manifest_bytes).hexdigest()
    if success_digest != actual_digest:
        raise CheckpointValidationError(
            f"Checkpoint commit digest mismatch in {directory}."
        )
    manifest = _read_json_object(manifest_path)
    schema_version = _validate_checkpoint_schema_version(
        manifest.get("schema_version"),
        location="checkpoint manifest",
    )
    iteration = _validate_checkpoint_iteration(
        manifest,
        schema_version,
        location="Checkpoint manifest",
    )
    manifest_completed_epochs = _validate_optional_completed_epochs(
        manifest,
        schema_version=schema_version,
        location="Checkpoint manifest",
    )
    world_size = manifest.get("world_size")
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        raise CheckpointValidationError("Checkpoint world_size must be positive.")
    has_ema = manifest.get("has_ema")
    has_scaler = manifest.get("has_scaler")
    if not isinstance(has_ema, bool) or not isinstance(has_scaler, bool):
        raise CheckpointValidationError(
            "Checkpoint has_ema and has_scaler fields must be booleans."
        )
    if require_ema and not has_ema:
        raise CheckpointValidationError(
            f"Checkpoint {directory} does not contain EMA state."
        )
    manifest_custom_roles = _validate_custom_checkpoint_roles(
        manifest,
        schema_version,
        has_ema=has_ema,
        location="Checkpoint manifest",
    )
    _inventory, roles = _validate_inventory(directory, manifest)
    _validate_state_roles(
        directory,
        roles,
        world_size=world_size,
        has_scaler=has_scaler,
        has_ema=has_ema,
        custom_checkpoint_roles=manifest_custom_roles,
    )

    state = _read_json_object(directory / "training_state.json")
    state_schema_version = _validate_checkpoint_schema_version(
        state.get("checkpoint_schema_version"),
        location="checkpoint training-state",
    )
    if state_schema_version != schema_version:
        raise CheckpointValidationError(
            f"Manifest/training_state schema mismatch in {directory}."
        )
    state_custom_roles = _validate_custom_checkpoint_roles(
        state,
        state_schema_version,
        has_ema=has_ema,
        location="Checkpoint training_state",
    )
    if state_custom_roles != manifest_custom_roles:
        raise CheckpointValidationError(
            f"Manifest/training_state custom-state role mismatch in {directory}."
        )
    state_iteration = _validate_checkpoint_iteration(
        state,
        state_schema_version,
        location="Checkpoint training_state",
    )
    state_completed_epochs = _validate_optional_completed_epochs(
        state,
        schema_version=state_schema_version,
        location="Checkpoint training_state",
    )
    if state_iteration != iteration:
        raise CheckpointValidationError(
            f"Manifest/training_state iteration mismatch in {directory}."
        )
    if state_completed_epochs != manifest_completed_epochs:
        raise CheckpointValidationError(
            f"Manifest/training_state completed_epochs mismatch in {directory}."
        )
    if state.get("experiment") != manifest.get("experiment"):
        raise CheckpointValidationError(
            f"Manifest/training_state experiment mismatch in {directory}."
        )
    if state.get("training_config") != manifest.get("training_config"):
        raise CheckpointValidationError(
            f"Manifest/training_state config mismatch in {directory}."
        )
    return iteration, state


def _validate_legacy_numbered_checkpoint(
    directory: Path,
    *,
    require_ema: bool,
) -> tuple[int, dict[str, Any]]:
    state = _read_json_object(directory / "training_state.json")
    if "checkpoint_schema_version" in state:
        raise CheckpointValidationError(
            "Legacy checkpoint training-state schema must be absent, found "
            f"{state.get('checkpoint_schema_version')!r}."
        )
    iteration = _validate_checkpoint_iteration(
        state,
        1,
        location="Legacy checkpoint training_state",
    )
    _validate_optional_completed_epochs(
        state,
        schema_version=1,
        location="Legacy checkpoint training_state",
    )
    config = state.get("training_config")
    if not isinstance(config, dict):
        raise CheckpointValidationError(
            f"Legacy checkpoint lacks training_config: {directory}."
        )
    world_size = config.get("world_size", 1)
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        raise CheckpointValidationError(
            f"Legacy checkpoint has invalid world_size: {directory}."
        )
    has_ema = (directory / "unet_ema").is_dir()
    if require_ema and not has_ema:
        raise CheckpointValidationError(
            f"Legacy checkpoint does not contain EMA weights: {directory}."
        )
    inventory = _build_inventory(directory)
    roles = _classify_accelerate_state(inventory)
    has_scaler = bool(roles["scaler"])
    _validate_state_roles(
        directory,
        roles,
        world_size=world_size,
        has_scaler=has_scaler,
        has_ema=has_ema,
    )
    return iteration, state


def _collective_raise_if_failed(
    accelerator,
    local_error: Exception | None,
    phase: str,
) -> None:
    failure = torch.tensor(
        1 if local_error is not None else 0,
        device=accelerator.device,
        dtype=torch.int64,
    )
    failure_count = int(accelerator.reduce(failure, reduction="sum").item())
    if failure_count:
        if local_error is None:
            raise RuntimeError(
                f"Checkpoint {phase} failed on {failure_count} other process(es)."
            )
        raise RuntimeError(f"Checkpoint {phase} failed: {local_error}") from local_error


def _acquire_writer_lock(checkpoint_root: Path):
    if fcntl is None:
        raise RuntimeError("Atomic checkpoint writing requires fcntl on this platform.")
    handle = (checkpoint_root / ".checkpoint.lock").open("a+b")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError(
            f"Another checkpoint writer is active below {checkpoint_root}."
        )
    return handle


def _release_writer_lock(handle) -> None:
    if handle is None:
        return
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _install_latest_pointer(
    checkpoint_root: Path,
    generation: Path,
    token: str,
) -> None:
    root_resolved = checkpoint_root.resolve()
    generation_resolved = generation.resolve(strict=True)
    if generation_resolved.parent != root_resolved:
        raise CheckpointValidationError(
            f"Latest checkpoint target must be a direct child of {checkpoint_root}."
        )
    temporary_pointer = checkpoint_root / f".latest.new-{token}"
    if temporary_pointer.exists() or temporary_pointer.is_symlink():
        temporary_pointer.unlink()
    os.symlink(generation_resolved.name, temporary_pointer)
    try:
        os.replace(temporary_pointer, checkpoint_root / "latest")
        _fsync_directory(checkpoint_root)
    finally:
        if temporary_pointer.exists() or temporary_pointer.is_symlink():
            temporary_pointer.unlink()


def _cleanup_after_commit(checkpoint_root: Path, current_generation: Path) -> None:
    current_resolved = current_generation.resolve(strict=True)
    for path in checkpoint_root.iterdir():
        if path.is_symlink():
            if path.name.startswith(".latest.new-"):
                path.unlink()
            continue
        if not path.is_dir():
            continue
        if path.resolve() == current_resolved:
            continue
        if (
            path.name.startswith(".generation-")
            or path.name.startswith(".staging-")
            or path.name.startswith("checkpoint-")
        ):
            shutil.rmtree(path)


def save_latest_accelerator_checkpoint(
    accelerator,
    model,
    ema,
    output_dir: Path | str,
    iteration: int,
    training_config: Mapping[str, Any],
    experiment: str,
) -> Path:
    """Transactionally replace the single logical ``checkpoints/latest`` state.

    Accelerate state is written by every rank to an immutable staging directory.
    Only after raw/EMA evaluation weights, semantic metadata, a complete file
    inventory, and the success marker exist does rank zero atomically swap the
    relative ``latest`` symlink. The old generation is removed only after that
    pointer commit succeeds.
    """

    if (
        isinstance(iteration, bool)
        or not isinstance(iteration, int)
        or iteration < 0
    ):
        raise ValueError("iteration must be a nonnegative integer.")
    if not isinstance(training_config, Mapping):
        raise TypeError("training_config must be a mapping.")
    if not isinstance(experiment, str) or not experiment.strip():
        raise ValueError("experiment must be a nonempty string.")

    output_directory = Path(output_dir).expanduser()
    checkpoint_root = output_directory / "checkpoints"
    lock_handle = None
    token = ""
    staging = checkpoint_root / ".staging-uninitialized"
    generation = checkpoint_root / ".generation-uninitialized"
    try:
        setup_error: Exception | None = None
        if accelerator.is_main_process:
            try:
                checkpoint_root.mkdir(parents=True, exist_ok=True)
                lock_handle = _acquire_writer_lock(checkpoint_root)
            except Exception as error:  # propagated collectively below
                setup_error = error
        _collective_raise_if_failed(accelerator, setup_error, "writer-lock setup")

        token_payload = [uuid.uuid4().hex if accelerator.is_main_process else None]
        broadcast_object_list(token_payload, from_process=0)
        token = str(token_payload[0])
        staging = checkpoint_root / f".staging-{token}"
        generation = checkpoint_root / f".generation-{token}"

        reserve_error: Exception | None = None
        if accelerator.is_main_process:
            try:
                staging.mkdir(parents=False, exist_ok=False)
            except Exception as error:
                reserve_error = error
        _collective_raise_if_failed(accelerator, reserve_error, "staging reservation")
        accelerator.wait_for_everyone()

        save_state_error: Exception | None = None
        try:
            accelerator.save_state(str(staging))
        except Exception as error:
            save_state_error = error
        _collective_raise_if_failed(accelerator, save_state_error, "Accelerate save_state")
        accelerator.wait_for_everyone()

        sidecar_error: Exception | None = None
        if accelerator.is_main_process:
            cpu_rng_state = torch.get_rng_state()
            cuda_rng_states = (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            )
            try:
                accelerator.unwrap_model(model).save_pretrained(
                    staging / "unet",
                    safe_serialization=True,
                )
                if ema is not None:
                    ema.save_pretrained(staging / "unet_ema")
                custom_checkpoint_roles = _expected_custom_checkpoint_roles(
                    has_ema=ema is not None,
                    schema_version=CHECKPOINT_SCHEMA_VERSION,
                )
                state = {
                    "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "experiment": experiment,
                    "global_iteration": int(iteration),
                    "weight_for_separate_evaluation": (
                        "unet_ema" if ema is not None else "unet"
                    ),
                    "custom_checkpoint_roles": custom_checkpoint_roles,
                    "training_config": dict(training_config),
                }
                _atomic_write_bytes(
                    staging / "training_state.json",
                    _canonical_json_bytes(state),
                )
                inventory = _build_inventory(staging)
                roles = _classify_accelerate_state(inventory)
                has_scaler = accelerator.scaler is not None
                _validate_state_roles(
                    staging,
                    roles,
                    world_size=int(accelerator.num_processes),
                    has_scaler=has_scaler,
                    has_ema=ema is not None,
                    custom_checkpoint_roles=custom_checkpoint_roles,
                )
                manifest = {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "experiment": experiment,
                    "global_iteration": int(iteration),
                    "training_config": dict(training_config),
                    "world_size": int(accelerator.num_processes),
                    "has_scaler": has_scaler,
                    "has_ema": ema is not None,
                    "custom_checkpoint_roles": custom_checkpoint_roles,
                    "versions": {
                        "torch": torch.__version__,
                        "accelerate": _package_version("accelerate"),
                        "diffusers": _package_version("diffusers"),
                    },
                    "accelerate_state_files": roles,
                    "inventory": inventory,
                }
                manifest_bytes = _canonical_json_bytes(manifest)
                _atomic_write_bytes(staging / MANIFEST_NAME, manifest_bytes)
                _atomic_write_bytes(
                    staging / SUCCESS_NAME,
                    (hashlib.sha256(manifest_bytes).hexdigest() + "\n").encode(
                        "ascii"
                    ),
                )
                _fsync_directory(staging)
            except Exception as error:
                sidecar_error = error
            finally:
                # EMAModel.save_pretrained constructs a model from config and
                # can consume torch RNG. Saving a checkpoint must not advance
                # rank zero's source/time stream relative to the other ranks or
                # to the RNG state that Accelerator just persisted.
                torch.set_rng_state(cpu_rng_state)
                if cuda_rng_states is not None:
                    torch.cuda.set_rng_state_all(cuda_rng_states)
        _collective_raise_if_failed(accelerator, sidecar_error, "manifest finalization")
        accelerator.wait_for_everyone()

        commit_error: Exception | None = None
        if accelerator.is_main_process:
            try:
                os.replace(staging, generation)
                _install_latest_pointer(checkpoint_root, generation, token)
            except Exception as error:
                commit_error = error
        _collective_raise_if_failed(accelerator, commit_error, "latest-pointer commit")
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            try:
                _cleanup_after_commit(checkpoint_root, generation)
            except Exception as error:
                warnings.warn(
                    f"The new latest checkpoint is committed, but old checkpoint "
                    f"cleanup failed: {error}",
                    RuntimeWarning,
                )
        accelerator.wait_for_everyone()
        return generation.resolve(strict=True)
    finally:
        if accelerator.is_main_process:
            _release_writer_lock(lock_handle)


def _candidate_iteration(
    path: Path,
    *,
    require_ema: bool,
    allow_legacy_numbered: bool,
) -> tuple[int, dict[str, Any]]:
    if (path / MANIFEST_NAME).exists() or (path / SUCCESS_NAME).exists():
        return _validate_committed_generation(path, require_ema=require_ema)
    if allow_legacy_numbered and path.name.startswith("checkpoint-"):
        return _validate_legacy_numbered_checkpoint(
            path,
            require_ema=require_ema,
        )
    raise CheckpointValidationError(f"Uncommitted checkpoint candidate: {path}.")


def validate_checkpoint_path(
    checkpoint: Path | str,
    *,
    require_ema: bool,
    allow_legacy_numbered: bool = True,
) -> Path:
    """Validate and canonicalize one explicitly selected checkpoint path."""

    if not isinstance(require_ema, bool):
        raise TypeError("require_ema must be a boolean.")
    if not isinstance(allow_legacy_numbered, bool):
        raise TypeError("allow_legacy_numbered must be a boolean.")
    requested = Path(checkpoint).expanduser()
    try:
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise FileNotFoundError(
            f"Checkpoint path does not exist: {requested}."
        ) from error
    if not resolved.is_dir():
        raise CheckpointValidationError(
            f"Checkpoint path is not a directory: {resolved}."
        )
    if requested.name.startswith(".staging-") or resolved.name.startswith(
        ".staging-"
    ):
        raise CheckpointValidationError(
            f"An uncommitted staging directory cannot be resumed: {resolved}."
        )
    _candidate_iteration(
        resolved,
        require_ema=require_ema,
        allow_legacy_numbered=allow_legacy_numbered,
    )
    return resolved


def resolve_checkpoint(
    output_dir: Path | str,
    require_ema: bool,
    allow_legacy_numbered: bool = True,
) -> Path:
    """Resolve and pin the committed latest checkpoint generation.

    A valid existing ``latest`` pointer always wins. If it is absent or broken,
    committed hidden generations are recoverable. Numbered checkpoints from the
    previous implementation can optionally be selected and atomically pinned by
    a relative ``latest`` symlink; the first subsequent transactional save then
    removes the legacy directories.
    """

    if not isinstance(require_ema, bool):
        raise TypeError("require_ema must be a boolean.")
    if not isinstance(allow_legacy_numbered, bool):
        raise TypeError("allow_legacy_numbered must be a boolean.")
    checkpoint_root = Path(output_dir).expanduser() / "checkpoints"
    latest = checkpoint_root / "latest"
    latest_error: CheckpointValidationError | None = None

    if latest.exists() and not latest.is_symlink():
        raise CheckpointValidationError(
            f"The logical latest checkpoint must be a symbolic link: {latest}."
        )
    if latest.exists() or latest.is_symlink():
        try:
            resolved = latest.resolve(strict=True)
            if resolved.parent != checkpoint_root.resolve():
                raise CheckpointValidationError(
                    f"Latest pointer escapes checkpoint root: {latest}."
                )
            _candidate_iteration(
                resolved,
                require_ema=require_ema,
                allow_legacy_numbered=allow_legacy_numbered,
            )
            return resolved
        except (OSError, CheckpointValidationError) as error:
            latest_error = (
                error
                if isinstance(error, CheckpointValidationError)
                else CheckpointValidationError(f"Broken latest pointer: {latest}.")
            )

    candidates: list[tuple[int, str, Path]] = []
    candidate_errors: list[str] = []
    if checkpoint_root.exists():
        candidate_paths = list(checkpoint_root.glob(".generation-*"))
        if allow_legacy_numbered:
            candidate_paths.extend(checkpoint_root.glob("checkpoint-*"))
        for path in candidate_paths:
            if not path.is_dir() or path.is_symlink():
                continue
            try:
                iteration, _state = _candidate_iteration(
                    path,
                    require_ema=require_ema,
                    allow_legacy_numbered=allow_legacy_numbered,
                )
                candidates.append(
                    (iteration, path.name, path.resolve(strict=True))
                )
            except CheckpointValidationError as error:
                candidate_errors.append(str(error))

    if not candidates:
        if latest_error is not None:
            raise latest_error
        details = f" Invalid candidates: {candidate_errors}" if candidate_errors else ""
        raise FileNotFoundError(
            f"No committed checkpoint was found below {checkpoint_root}.{details}"
        )

    _iteration, _name, selected = max(
        candidates, key=lambda item: (item[0], item[1])
    )
    if latest.exists() and not latest.is_symlink():
        raise CheckpointValidationError(
            f"Cannot recover latest because a non-symlink path occupies {latest}."
        )
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    _install_latest_pointer(checkpoint_root, selected, uuid.uuid4().hex)
    return selected.resolve(strict=True)
