#!/usr/bin/env python3
"""Build an immutable golden-v0 artifact manifest for the riskml-capstone research repository.

This script is a READ-ONLY auditor. It reads existing on-disk artifacts, computes
integrity and content hashes, records schema contracts and full provenance, and writes
a single manifest JSON. It NEVER executes notebooks, regenerates artifacts, or modifies
any input file.

golden-v0 is the immutable evidence snapshot of the SUBMITTED capstone state. It is not a
claim that the pipeline was recomputed on the freeze date.

Usage (run from the repository root, inside the capstone conda environment):

    python scripts/build_golden_manifest.py \
        --artifact-root artifacts/golden-v0 \
        --manifest-path artifacts/golden-v0/golden-v0.manifest.json
"""

# ==== IMPORT ARGPARSE FOR COMMAND-LINE ARGUMENT PARSING ====
import argparse

# ==== IMPORT HASHLIB FOR SHA256 DIGESTS ====
import hashlib

# ==== IMPORT JSON FOR STABLE MANIFEST SERIALIZATION ====
import json

# ==== IMPORT PLATFORM FOR OPERATING-SYSTEM METADATA ====
import platform

# ==== IMPORT SUBPROCESS FOR GIT AND PIP COMMANDS ====
import subprocess

# ==== IMPORT SYS FOR PYTHON EXECUTABLE AND VERSION METADATA ====
import sys

# ==== IMPORT DATETIME FOR UTC TIMESTAMPS ====
from datetime import datetime, timezone

# ==== IMPORT PACKAGE-VERSION LOOKUP FROM IMPORTLIB METADATA ====
from importlib.metadata import PackageNotFoundError, version

# ==== IMPORT PATH FOR FILESYSTEM PATH HANDLING ====
from pathlib import Path

# ==== IMPORT NUMPY FOR STABLE UINT64 BYTE NORMALIZATION ====
import numpy as np

# ==== IMPORT PANDAS FOR DATAFRAME LOADING AND HASHING ====
import pandas as pd


# ============================================================================
# Provenance helpers
# ============================================================================

# ==== RETURN A TIMEZONE-AWARE UTC ISO-8601 TIMESTAMP ====
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ==== RUN A SHELL COMMAND AND RETURN STRIPPED STANDARD OUTPUT ====
def run_command(command: list) -> str:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


# ==== RUN A SHELL COMMAND BUT RETURN AN EMPTY STRING ON FAILURE ====
def run_command_or_empty(command: list) -> str:
    try:
        return run_command(command)
    except Exception:
        return ""


# ==== RETURN AN INSTALLED PACKAGE VERSION OR A MISSING MARKER ====
def package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "NOT_INSTALLED"


# ============================================================================
# Hashing helpers
# ============================================================================

# ==== COMPUTE A BYTE-LEVEL SHA256 HASH OF A FILE (FILE-INTEGRITY PROOF) ====
def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ==== COMPUTE A SHA256 HASH OF A JSON-SERIALIZABLE OBJECT WITH STABLE KEY ORDER ====
def json_sha256(payload) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# ==== BUILD A DATAFRAME CONTRACT (SHAPE, INDEX, COLUMNS, DTYPES, NULL COUNTS) ====
# NOTE: rows and columns are NEVER sorted; order is part of the research contract.
def dataframe_contract(dataframe: pd.DataFrame) -> dict:
    return {
        "shape": [int(dataframe.shape[0]), int(dataframe.shape[1])],
        "index_class": type(dataframe.index).__name__,
        "index_names": [str(name) for name in dataframe.index.names],
        "index_dtype": str(getattr(dataframe.index, "dtype", "MULTIINDEX")),
        "index_start": str(dataframe.index[0]) if len(dataframe.index) else None,
        "index_end": str(dataframe.index[-1]) if len(dataframe.index) else None,
        "index_is_monotonic_increasing": bool(dataframe.index.is_monotonic_increasing),
        "index_duplicate_count": int(dataframe.index.duplicated().sum()),
        "columns": [str(col) for col in dataframe.columns],
        "dtypes": {str(col): str(dataframe[col].dtype) for col in dataframe.columns},
        "null_counts": {str(col): int(dataframe[col].isna().sum()) for col in dataframe.columns},
    }


# ==== COMPUTE A DATAFRAME CONTENT HASH (SCIENTIFIC-EQUIVALENCE PROOF) ====
def dataframe_content_sha256(dataframe: pd.DataFrame) -> str:
    contract_payload = dataframe_contract(dataframe)
    row_hashes = pd.util.hash_pandas_object(dataframe, index=True, categorize=False)
    row_hash_bytes = row_hashes.to_numpy(dtype=np.uint64).astype("<u8", copy=False).tobytes()
    contract_bytes = json.dumps(contract_payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(contract_bytes)
    digest.update(row_hash_bytes)
    return digest.hexdigest()


# ==== COMPUTE A NAN-MASK HASH (ALIGNMENT / LEAKAGE-DRIFT PROOF) ====
def dataframe_nan_mask_sha256(dataframe: pd.DataFrame) -> str:
    nan_mask = dataframe.isna()
    mask_hashes = pd.util.hash_pandas_object(nan_mask, index=True, categorize=False)
    mask_hash_bytes = mask_hashes.to_numpy(dtype=np.uint64).astype("<u8", copy=False).tobytes()
    return hashlib.sha256(mask_hash_bytes).hexdigest()


# ============================================================================
# Artifact loading
# ============================================================================

# ==== LOAD A PARQUET OR CSV ARTIFACT AS A DATAFRAME, OR RETURN NONE ====
def load_dataframe(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        # Read CSV without forcing an index so the contract reflects the file as written.
        return pd.read_csv(path)
    return None


# ==== BUILD ONE ARTIFACT RECORD (HASHES + CONTRACT WHERE APPLICABLE) ====
def build_artifact_record(path: Path, root: Path) -> dict:
    relative_path = path.relative_to(root).as_posix()
    stat = path.stat()
    record = {
        "path": relative_path,
        "suffix": path.suffix.lower(),
        "file_size_bytes": int(stat.st_size),
        "file_modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "file_sha256": file_sha256(path),
    }
    # Add dataframe contracts and content hashes for parquet/csv; leave JSON/other as byte-hash only.
    try:
        dataframe = load_dataframe(path)
    except Exception as exc:
        record["load_error"] = f"{type(exc).__name__}: {exc}"
        dataframe = None
    if dataframe is not None:
        contract_payload = dataframe_contract(dataframe)
        record.update({
            "dataframe_contract": contract_payload,
            "schema_sha256": json_sha256(contract_payload),
            "content_sha256": dataframe_content_sha256(dataframe),
            "nan_mask_sha256": dataframe_nan_mask_sha256(dataframe),
        })
    return record


# ============================================================================
# Snapshot-level metadata
# ============================================================================

# ==== BUILD SNAPSHOT IDENTITY, GIT PROVENANCE, ENVIRONMENT, AND DETERMINISM METADATA ====
def build_snapshot_metadata(root: Path) -> dict:
    return {
        "golden_version": "golden-v0",
        "snapshot_kind": "immutable-evidence-snapshot-of-submitted-capstone",
        "snapshot_note": (
            "golden-v0 is the immutable artifact snapshot of the submitted capstone state. "
            "It is NOT a claim that the pipeline was recomputed on the freeze date."
        ),
        "created_at_utc": utc_now_iso(),
        "created_by": "Steven Archuleta",
        "approval_reason": "initial-golden-freeze",
        "manifest_schema_version": "1.0",
        "artifact_root": root.as_posix(),
        # ---- Git provenance ----
        "git_remote": run_command_or_empty(["git", "config", "--get", "remote.origin.url"]),
        "git_branch": run_command_or_empty(["git", "branch", "--show-current"]),
        "git_head_sha": run_command_or_empty(["git", "rev-parse", "HEAD"]),
        "git_status_porcelain": run_command_or_empty(["git", "status", "--porcelain"]),
        # ---- Environment ----
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "pip_freeze": run_command_or_empty([sys.executable, "-m", "pip", "freeze"]),
        "package_versions": {
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "pyarrow": package_version("pyarrow"),
            "scikit-learn": package_version("scikit-learn"),
            "xgboost": package_version("xgboost"),
            "statsmodels": package_version("statsmodels"),
            "matplotlib": package_version("matplotlib"),
            "streamlit": package_version("streamlit"),
        },
        # ---- Determinism controls (recorded for provenance; project seed is 692) ----
        "determinism": {
            "project_seed": 692,
            "pythonhashseed_env": run_command_or_empty([sys.executable, "-c", "import os;print(os.environ.get('PYTHONHASHSEED',''))"]),
        },
        # ---- Method constants (the research contract) ----
        "method_constants": {
            "forecast_horizon_days": 20,
            "annualization_factor": 252,
            "ddof": 1,
            "target_construction": "two-step shift, ddof=1, 20-day forward realized volatility",
            "dag_allowed_prefixes": ["VOL__", "MACRO__", "REGIME__"],
        },
    }


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Build a riskml golden-v0 artifact manifest (read-only).")
    parser.add_argument("--artifact-root", required=True, help="Root directory containing the golden-v0 artifact copies.")
    parser.add_argument("--manifest-path", required=True, help="Path where the manifest JSON will be written.")
    args = parser.parse_args()

    artifact_root = Path(args.artifact_root).resolve()
    manifest_path = Path(args.manifest_path).resolve()

    if not artifact_root.exists():
        raise SystemExit(f"Artifact root does not exist: {artifact_root}")

    # Collect supported artifacts: parquet, csv (dataframe-contracted) and json (byte-hash only).
    artifact_paths = sorted(
        list(artifact_root.rglob("*.parquet"))
        + list(artifact_root.rglob("*.csv"))
        + list(artifact_root.rglob("*.json"))
    )
    # Exclude the manifest file itself if it already lives under the artifact root.
    artifact_paths = [p for p in artifact_paths if p.resolve() != manifest_path]

    manifest = build_snapshot_metadata(artifact_root)
    manifest["artifact_count"] = len(artifact_paths)
    manifest["artifacts"] = [build_artifact_record(path, artifact_root) for path in artifact_paths]

    # Seal the manifest with its own content hash (computed over everything above).
    manifest["manifest_content_sha256"] = json_sha256(manifest)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Manifest written: {manifest_path}")
    print(f"Artifacts recorded: {len(artifact_paths)}")
    print(f"Manifest content SHA-256: {manifest['manifest_content_sha256']}")


if __name__ == "__main__":
    main()
