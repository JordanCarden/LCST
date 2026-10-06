from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
from sklearn.decomposition import PCA


MOLFORMER_MODEL_ID = "ibm-research/MoLFormer-XL-both-10pct"
MOLFORMER_SNAPSHOT_REVISION = "7b12d946c181a37f6012b9dc3b002275de070314"
DEFAULT_MOLFORMER_SNAPSHOT = (
    Path(HF_HUB_CACHE)
    / "models--ibm-research--MoLFormer-XL-both-10pct"
    / "snapshots"
    / MOLFORMER_SNAPSHOT_REVISION
)
MOLFORMER_SNAPSHOT_FILES = (
    "config.json",
    "configuration_molformer.py",
    "model.safetensors",
    "modeling_molformer.py",
    "special_tokens_map.json",
    "tokenization_molformer.py",
    "tokenization_molformer_fast.py",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
PCA_COMPONENTS = 16


@dataclass(frozen=True)
class MolFormerPCAResult:
    pca: PCA
    scores: dict[str, list[float]]
    fit_chemical_names: tuple[str, ...]
    transformed_chemical_names: tuple[str, ...]
    provenance: dict[str, Any]


def _ordered_unique(names: Iterable[str]) -> tuple[str, ...]:
    ordered = tuple(str(name) for name in names)
    if not ordered:
        raise ValueError("At least one chemical name is required")
    if len(set(ordered)) != len(ordered):
        duplicates = sorted({name for name in ordered if ordered.count(name) > 1})
        raise ValueError(f"Chemical list contains duplicates: {duplicates}")
    return ordered


def validate_structure_registry(
    chemical_metadata: Mapping[str, Any],
    *,
    expected_names: Iterable[str] | None = None,
) -> dict[str, dict[str, str]]:
    molformer = chemical_metadata.get("molformer")
    if not isinstance(molformer, Mapping):
        raise ValueError("chemical_metadata.molformer is missing")
    if molformer.get("model_id") != MOLFORMER_MODEL_ID:
        raise ValueError(f"Expected MoLFormer model {MOLFORMER_MODEL_ID!r}")
    if molformer.get("snapshot_revision") != MOLFORMER_SNAPSHOT_REVISION:
        raise ValueError(f"Expected MoLFormer snapshot {MOLFORMER_SNAPSHOT_REVISION!r}")
    raw_structures = molformer.get("structures")
    if not isinstance(raw_structures, Mapping) or not raw_structures:
        raise ValueError("chemical_metadata.molformer.structures is missing or empty")

    structures: dict[str, dict[str, str]] = {}
    for raw_name, raw_record in raw_structures.items():
        name = str(raw_name).strip()
        if not name or not isinstance(raw_record, Mapping):
            raise ValueError("Every chemical structure requires a nonempty name and metadata record")
        smiles = str(raw_record.get("canonical_smiles", "")).strip()
        role = str(raw_record.get("role", "")).strip()
        source = str(raw_record.get("source", "")).strip()
        if not smiles:
            raise ValueError(f"Chemical {name!r} has no molecular input string")
        if role not in {"additive", "salt"}:
            raise ValueError(f"Chemical {name!r} has invalid role {role!r}")
        structures[name] = {
            "canonical_smiles": smiles,
            "role": role,
            "source": source,
        }

    if expected_names is not None:
        expected = set(_ordered_unique(expected_names))
        actual = set(structures)
        if actual != expected:
            raise ValueError(
                "MoLFormer structure registry does not match the expected chemicals; "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
    return structures


def structure_registry_sha256(structures: Mapping[str, Mapping[str, str]]) -> str:
    payload = {
        name: {
            "canonical_smiles": structures[name]["canonical_smiles"],
            "role": structures[name]["role"],
        }
        for name in sorted(structures)
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def pooled_molformer_embedding(outputs: object, attention_mask: Any) -> Any:
    pooler_output = getattr(outputs, "pooler_output", None)
    if pooler_output is not None:
        return pooler_output
    last_hidden = getattr(outputs, "last_hidden_state", None)
    if last_hidden is None:
        raise RuntimeError("MoLFormer output has neither pooler_output nor last_hidden_state")
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


def resolve_molformer_snapshot(snapshot_path: Path | None = None) -> Path:
    """Reuse the pinned cache offline, downloading missing files on first use.

    HF_HOME/HF_HUB_CACHE control the cache location. An explicit snapshot path
    or LCST_MOLFORMER_SNAPSHOT remains a local-only override.
    """
    configured_path = (
        snapshot_path if snapshot_path is not None else os.environ.get("LCST_MOLFORMER_SNAPSHOT")
    )
    if configured_path is not None:
        path = Path(configured_path).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"Configured MoLFormer snapshot not found: {path}")
        return path

    if all((DEFAULT_MOLFORMER_SNAPSHOT / name).is_file() for name in MOLFORMER_SNAPSHOT_FILES):
        return DEFAULT_MOLFORMER_SNAPSHOT

    print(
        f"Downloading MoLFormer revision {MOLFORMER_SNAPSHOT_REVISION} to the local Hugging Face cache...",
        file=sys.stderr, flush=True,
    )
    try:
        path = Path(snapshot_download(
            repo_id=MOLFORMER_MODEL_ID,
            revision=MOLFORMER_SNAPSHOT_REVISION,
            cache_dir=HF_HUB_CACHE,
            allow_patterns=list(MOLFORMER_SNAPSHOT_FILES),
        ))
    except Exception as error:
        raise RuntimeError(
            "The study's MoLFormer model is not fully cached and could not be downloaded. "
            "Connect to the internet for the first run, or set LCST_MOLFORMER_SNAPSHOT "
            f"to an existing local copy of revision {MOLFORMER_SNAPSHOT_REVISION}."
        ) from error
    missing = [name for name in MOLFORMER_SNAPSHOT_FILES if not (path / name).is_file()]
    if missing:
        raise RuntimeError(f"Incomplete MoLFormer download at {path}; missing files: {missing}")
    return path


def generate_molformer_embeddings(
    structures: Mapping[str, Mapping[str, str]],
    *,
    chemical_names: Iterable[str] | None = None,
    snapshot_path: Path | None = None,
    device: str | None = None,
) -> dict[str, np.ndarray]:
    names = _ordered_unique(chemical_names if chemical_names is not None else sorted(structures))
    unknown = sorted(set(names) - set(structures))
    if unknown:
        raise ValueError(f"No molecular input is registered for: {unknown}")
    os.environ.setdefault("HF_MODULES_CACHE", "/tmp/lcst_hf_modules")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    if not str(transformers.__version__).startswith("4."):
        raise RuntimeError(f"MoLFormer was validated with transformers 4.x, not {transformers.__version__}")
    snapshot_path = resolve_molformer_snapshot(snapshot_path)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    encoder = AutoModel.from_pretrained(
        snapshot_path,
        deterministic_eval=True,
        trust_remote_code=True,
        local_files_only=True,
    )
    encoder.eval()
    encoder.to(selected_device)

    embeddings: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for name in names:
            inputs = tokenizer(
                [structures[name]["canonical_smiles"]],
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            inputs = {key: value.to(selected_device) for key, value in inputs.items()}
            output = encoder(**inputs)
            pooled = pooled_molformer_embedding(output, inputs["attention_mask"])
            vector = pooled.detach().cpu().numpy().astype(np.float32).reshape(-1)
            if vector.size == 0 or not np.isfinite(vector).all():
                raise ValueError(f"MoLFormer produced an invalid embedding for {name!r}")
            embeddings[name] = vector

    dimensions = {vector.shape for vector in embeddings.values()}
    if len(dimensions) != 1:
        raise ValueError(f"MoLFormer embedding dimensions are inconsistent: {sorted(dimensions)}")
    return embeddings


def fit_molformer_pca(
    embeddings: Mapping[str, np.ndarray],
    *,
    fit_chemical_names: Iterable[str],
    transform_chemical_names: Iterable[str] | None = None,
    n_components: int = PCA_COMPONENTS,
    structure_sha256: str | None = None,
) -> MolFormerPCAResult:
    import sklearn
    import torch
    import transformers

    fit_names = _ordered_unique(fit_chemical_names)
    transform_names = _ordered_unique(
        transform_chemical_names if transform_chemical_names is not None else fit_names
    )
    required = set(fit_names) | set(transform_names)
    missing = sorted(required - set(embeddings))
    if missing:
        raise ValueError(f"Missing MoLFormer embeddings for: {missing}")
    if len(fit_names) < n_components + 1:
        raise ValueError(
            f"PCA-{n_components} requires at least {n_components + 1} fitting chemicals; "
            f"received {len(fit_names)}"
        )

    fit_matrix = np.vstack([np.asarray(embeddings[name], dtype=float) for name in fit_names])
    transform_matrix = np.vstack([np.asarray(embeddings[name], dtype=float) for name in transform_names])
    if fit_matrix.ndim != 2 or transform_matrix.shape[1] != fit_matrix.shape[1]:
        raise ValueError("MoLFormer embeddings do not share one feature dimension")
    if not np.isfinite(fit_matrix).all() or not np.isfinite(transform_matrix).all():
        raise ValueError("MoLFormer embeddings contain nonfinite values")

    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(fit_matrix)
    transformed = pca.transform(transform_matrix)
    if transformed.shape != (len(transform_names), n_components) or not np.isfinite(transformed).all():
        raise ValueError("PCA transformation produced invalid scores")
    scores = {name: transformed[index].astype(float).tolist() for index, name in enumerate(transform_names)}
    provenance = {
        "model_id": MOLFORMER_MODEL_ID,
        "snapshot_revision": MOLFORMER_SNAPSHOT_REVISION,
        "structure_registry_sha256": structure_sha256,
        "fit_chemical_names": list(fit_names),
        "transformed_chemical_names": list(transform_names),
        "n_components": int(n_components),
        "embedding_dimension": int(fit_matrix.shape[1]),
        "software": {
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "pca_mean_sha256": hashlib.sha256(np.asarray(pca.mean_, dtype=np.float64).tobytes()).hexdigest(),
        "pca_components_sha256": hashlib.sha256(
            np.asarray(pca.components_, dtype=np.float64).tobytes()
        ).hexdigest(),
    }
    return MolFormerPCAResult(
        pca=pca,
        scores=scores,
        fit_chemical_names=fit_names,
        transformed_chemical_names=transform_names,
        provenance=provenance,
    )
