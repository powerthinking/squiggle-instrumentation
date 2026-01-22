from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from squiggle_core import paths
from squiggle_core.geometry.state import compute_effective_rank
from squiggle_core.schemas.probe_summar import (
    CaptureNotes,
    ProbeCaptureRef,
    ProbeIdentity,
    ProbeSummary,
    Ranking,
    RankingComponents,
    Signature,
)
from squiggle_core.schemas.probe_tables import probe_captures_index_schema


class ProbeAdapter(Protocol):
    def capture_layer_tensors(self, *, layers: List[int], device: str) -> Dict[int, torch.Tensor]:
        ...

    def micro_finetune(self, *, family_id: str, steps: int, lr: float, seed: int, device: str) -> None:
        ...


@dataclass
class DummyAdapter:
    d_model: int = 128
    seq_len: int = 32
    batch_size: int = 64

    def __post_init__(self) -> None:
        torch.manual_seed(0)
        self._layers: torch.nn.ModuleList = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(self.d_model, self.d_model),
                    torch.nn.Tanh(),
                )
                for _ in range(32)
            ]
        )

    def capture_layer_tensors(self, *, layers: List[int], device: str) -> Dict[int, torch.Tensor]:
        x = torch.randn(self.batch_size, self.seq_len, self.d_model, device=device)
        out: Dict[int, torch.Tensor] = {}
        for i, block in enumerate(self._layers):
            x = block(x)
            if i in layers:
                out[int(i)] = x.detach().clone()
        return out

    def micro_finetune(self, *, family_id: str, steps: int, lr: float, seed: int, device: str) -> None:
        torch.manual_seed(seed)
        opt = torch.optim.AdamW(self._layers.parameters(), lr=lr)
        for _ in range(int(steps)):
            x = torch.randn(self.batch_size, self.seq_len, self.d_model, device=device)
            y = x
            for block in self._layers:
                y = block(y)
            loss = (y**2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _sha256_json(d: dict) -> str:
    b = json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _sv_entropy(x: torch.Tensor) -> float:
    X = x
    if X.ndim == 3:
        b, t, d = X.shape
        X = X.reshape(b * t, d)
    if X.ndim == 2:
        pass
    elif X.ndim == 1:
        X = X.unsqueeze(0)
    else:
        raise ValueError(f"Unsupported tensor shape: {tuple(X.shape)}")

    X = X.float()
    X = X - X.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(X)
    eps = 1e-12
    s = torch.clamp(s, min=eps)
    p = s / s.sum()
    ent = -(p * torch.log(p)).sum().item()
    return float(ent)


def _sparsity_proxy(x: torch.Tensor, eps: float = 1e-3) -> float:
    X = x.float()
    return float((X.abs() < eps).float().mean().item())


def _principal_angle_mean(pre: torch.Tensor, post: torch.Tensor, k: int = 8) -> float:
    def to2d(z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 3:
            b, t, d = z.shape
            return z.reshape(b * t, d).float()
        if z.ndim == 2:
            return z.float()
        if z.ndim == 1:
            return z.unsqueeze(0).float()
        raise ValueError(f"Unsupported tensor shape: {tuple(z.shape)}")

    A = to2d(pre)
    B = to2d(post)

    A = A - A.mean(dim=0, keepdim=True)
    B = B - B.mean(dim=0, keepdim=True)

    _, _, vha = torch.linalg.svd(A, full_matrices=False)
    _, _, vhb = torch.linalg.svd(B, full_matrices=False)

    k = int(max(1, min(k, vha.shape[0], vhb.shape[0])))
    Va = vha[:k].T
    Vb = vhb[:k].T

    M = Va.T @ Vb
    s = torch.linalg.svdvals(M)
    s = torch.clamp(s, 0.0, 1.0)
    angles = torch.acos(s)
    return float(angles.mean().item())


def _write_probe_captures(
    run_id: str,
    probe_name: str,
    step: int,
    layer_tensors: Dict[int, torch.Tensor],
) -> List[Tuple[int, Path]]:
    out_dir = paths.probe_captures_dir(run_id, probe_name) / f"step_{step:06d}"
    _ensure_dir(out_dir)

    written: List[Tuple[int, Path]] = []
    for layer, t in sorted(layer_tensors.items()):
        fname = f"resid_layer_{layer:02d}.pt"
        p = out_dir / fname
        torch.save(t.detach().cpu(), p)
        written.append((int(layer), p))

    return written


def _write_probe_index(
    run_id: str,
    probe_name: str,
    probe_config_hash: str,
    records: List[Tuple[int, int, Path]],
) -> Path:
    created_at_utc = _utc_now()

    rows = []
    for step, layer, path in records:
        rel = path.relative_to(paths.runs_root()).as_posix()
        rows.append(
            {
                "run_id": run_id,
                "probe_name": probe_name,
                "probe_config_hash": probe_config_hash,
                "schema_version": "probe_captures_index@0.1",
                "capture_type": "activations_snapshot",
                "step": int(step),
                "shard_id": f"step_{step:06d}_layer_{layer:02d}",
                "path": rel,
                "bytes": int(path.stat().st_size),
                "checksum": "",
                "created_at_utc": created_at_utc,
            }
        )

    table = pa.Table.from_pylist(rows, schema=probe_captures_index_schema)
    out_path = paths.probe_index_path(run_id, probe_name)
    _ensure_dir(out_path.parent)
    pq.write_table(table, out_path.as_posix(), compression="zstd")
    return out_path


def _write_probe_manifest(
    run_id: str,
    probe_name: str,
    probe_config_hash: str,
    capture_steps_used: List[int],
) -> Path:
    out_path = paths.probe_manifest_path(run_id, probe_name)
    _ensure_dir(out_path.parent)

    payload = {
        "schema_version": "probe_manifest@0.1",
        "created_at_utc": _utc_now().isoformat(),
        "run_id": run_id,
        "probe_name": probe_name,
        "probe_config_hash": probe_config_hash,
        "capture_steps_used": sorted(set(int(x) for x in capture_steps_used)),
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def _write_meta_json(
    run_id: str,
    *,
    exp_id: str,
    family_id: str,
    model_id: str,
    base_checkpoint: str,
    steps: int,
    lr: float,
    seed: int,
    device: str,
    dtype: str,
    probe_name: str,
    layers_covered: List[int],
) -> Path:
    run_dir = paths.run_dir(run_id)
    _ensure_dir(run_dir)

    meta_path = run_dir / "meta.json"
    payload = {
        "run_id": run_id,
        "run_type": "probe_micro_finetune",
        "exp_id": exp_id,
        "family_id": family_id,
        "model_id": model_id,
        "base_checkpoint": base_checkpoint,
        "steps": int(steps),
        "lr": float(lr),
        "seed": int(seed),
        "device": device,
        "dtype": dtype,
        "probe_name": probe_name,
        "layers_covered": [int(x) for x in layers_covered],
        "created_at_utc": _utc_now().isoformat(),
    }
    meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return meta_path


def _make_run_id(
    *,
    exp_id: str,
    family_id: str,
    model_id: str,
    base_checkpoint: str,
    steps: int,
    seed: int,
    probe_config_hash: str,
) -> str:
    spec = {
        "exp_id": exp_id,
        "family_id": family_id,
        "model_id": model_id,
        "base_checkpoint": base_checkpoint,
        "steps": int(steps),
        "seed": int(seed),
        "probe_config_hash": probe_config_hash,
    }
    return "probe_" + _sha256_json(spec)[:16]


def run_probe_experiment(
    *,
    exp_id: str,
    model_id: str,
    base_checkpoint: str,
    family_ids: List[str],
    steps: int,
    lr: float,
    seed: int,
    device: str,
    dtype: str,
    probe_name: str,
    layers_covered: List[int],
    summaries_dir: Optional[str] = None,
    adapter: Optional[ProbeAdapter] = None,
) -> Path:
    if adapter is None:
        adapter = DummyAdapter()

    if summaries_dir is None:
        out = paths.data_root() / "experiments" / exp_id / "summaries"
    else:
        out = Path(summaries_dir).expanduser().resolve()
    _ensure_dir(out)

    capture_steps_used = [0, int(steps)]
    probe_config = {
        "probe_name": probe_name,
        "layers_covered": [int(x) for x in layers_covered],
        "capture_steps_used": capture_steps_used,
        "capture_type": "activations_snapshot",
    }
    probe_config_hash = _sha256_json(probe_config)[:16]

    analysis_spec = {
        "exp_id": exp_id,
        "probe_name": probe_name,
        "probe_config_hash": probe_config_hash,
        "signature_version": "sig@2.0",
        "score_version": "dis@1.0",
    }
    analysis_id = "analysis_" + _sha256_json(analysis_spec)[:16]

    for family_id in family_ids:
        run_id = _make_run_id(
            exp_id=exp_id,
            family_id=family_id,
            model_id=model_id,
            base_checkpoint=base_checkpoint,
            steps=steps,
            seed=seed,
            probe_config_hash=probe_config_hash,
        )

        _write_meta_json(
            run_id,
            exp_id=exp_id,
            family_id=family_id,
            model_id=model_id,
            base_checkpoint=base_checkpoint,
            steps=steps,
            lr=lr,
            seed=seed,
            device=device,
            dtype=dtype,
            probe_name=probe_name,
            layers_covered=layers_covered,
        )

        pre = adapter.capture_layer_tensors(layers=layers_covered, device=device)
        adapter.micro_finetune(family_id=family_id, steps=steps, lr=lr, seed=seed, device=device)
        post = adapter.capture_layer_tensors(layers=layers_covered, device=device)

        written_pre = _write_probe_captures(run_id, probe_name, 0, pre)
        written_post = _write_probe_captures(run_id, probe_name, steps, post)

        index_records: List[Tuple[int, int, Path]] = []
        for layer, p in written_pre:
            index_records.append((0, layer, p))
        for layer, p in written_post:
            index_records.append((steps, layer, p))

        manifest_path = _write_probe_manifest(run_id, probe_name, probe_config_hash, capture_steps_used)
        index_path = _write_probe_index(run_id, probe_name, probe_config_hash, index_records)

        eff_pre: List[float] = []
        eff_post: List[float] = []
        eff_delta: List[float] = []

        ent_pre: List[float] = []
        ent_post: List[float] = []
        ent_delta: List[float] = []

        sp_pre: List[float] = []
        sp_post: List[float] = []
        sp_delta: List[float] = []

        ang_post_vs_pre: List[float] = []

        for layer in layers_covered:
            x0 = pre[int(layer)]
            x1 = post[int(layer)]

            r0 = compute_effective_rank(x0)
            r1 = compute_effective_rank(x1)

            e0 = _sv_entropy(x0)
            e1 = _sv_entropy(x1)

            s0 = _sparsity_proxy(x0)
            s1 = _sparsity_proxy(x1)

            a = _principal_angle_mean(x0, x1)

            eff_pre.append(float(r0))
            eff_post.append(float(r1))
            eff_delta.append(float(r1 - r0))

            ent_pre.append(float(e0))
            ent_post.append(float(e1))
            ent_delta.append(float(e1 - e0))

            sp_pre.append(float(s0))
            sp_post.append(float(s1))
            sp_delta.append(float(s1 - s0))

            ang_post_vs_pre.append(float(a))

        drift_velocity_by_layer = [abs(x) / float(max(1, steps)) for x in eff_delta]
        alignment_velocity_by_layer = [abs(x) / float(max(1, steps)) for x in ang_post_vs_pre]
        drift_accel_by_layer = [0.0 for _ in layers_covered]
        volatility_by_layer = [0.0 for _ in layers_covered]

        drift_velocity_global = float(sum(drift_velocity_by_layer) / max(1, len(drift_velocity_by_layer)))
        alignment_velocity_global = float(sum(alignment_velocity_by_layer) / max(1, len(alignment_velocity_by_layer)))
        drift_accel_global = 0.0
        volatility_global = 0.0

        k = min(6, len(layers_covered))
        top_idx = sorted(range(len(drift_velocity_by_layer)), key=lambda i: -abs(drift_velocity_by_layer[i]))[:k]
        affected_layers = [int(layers_covered[i]) for i in top_idx]

        sig_vec = [
            float(sum(abs(x) for x in eff_delta) / max(1, len(eff_delta))),
            float(sum(abs(x) for x in ent_delta) / max(1, len(ent_delta))),
            float(drift_velocity_global),
            float(alignment_velocity_global),
        ]
        sig_norm = float(torch.tensor(sig_vec, dtype=torch.float32).norm(p=2).item())

        raw_mag = float(sum(abs(x) for x in sig_vec) / max(1, len(sig_vec)))
        magnitude = float(1.0 - torch.exp(torch.tensor(-raw_mag)).item())

        ps = ProbeSummary(
            analysis_id=analysis_id,
            identity=ProbeIdentity(
                family_id=family_id,
                model_id=model_id,
                base_checkpoint=base_checkpoint,
                run_mode="probe_micro_finetune",
                steps=int(steps),
                seed=int(seed),
                dtype=dtype,
                device=device,
                timestamp_utc=_utc_now(),
            ),
            capture_ref=ProbeCaptureRef(
                run_id=run_id,
                probe_name=probe_name,
                probe_config_hash=probe_config_hash,
                captures_manifest_path=str(paths.probe_manifest_path(run_id, probe_name).relative_to(paths.data_root())),
                captures_index_path=str(paths.probe_index_path(run_id, probe_name).relative_to(paths.data_root())),
                capture_steps_used=capture_steps_used,
                notes=CaptureNotes(dropped_steps=[], warnings=[]),
            ),
            layer_A_state={
                "description": "Basis-invariant geometric state descriptors (snapshot geometry).",
                "layers_covered": [int(x) for x in layers_covered],
                "metrics": {
                    "effective_rank": {"pre": eff_pre, "post": eff_post, "delta": eff_delta},
                    "sv_entropy": {"pre": ent_pre, "post": ent_post, "delta": ent_delta},
                    "sparsity_proxy": {"pre": sp_pre, "post": sp_post, "delta": sp_delta},
                    "principal_angle_post_vs_pre": ang_post_vs_pre,
                },
                "aggregations": {
                    "summary_pre": {
                        "mean_abs": float(sum(abs(x) for x in eff_pre) / max(1, len(eff_pre))),
                        "median_abs": float(sorted(abs(x) for x in eff_pre)[len(eff_pre) // 2]),
                        "p95_abs": float(sorted(abs(x) for x in eff_pre)[int(0.95 * (len(eff_pre) - 1))]),
                    },
                    "summary_delta": {
                        "mean_abs": float(sum(abs(x) for x in eff_delta) / max(1, len(eff_delta))),
                        "median_abs": float(sorted(abs(x) for x in eff_delta)[len(eff_delta) // 2]),
                        "p95_abs": float(sorted(abs(x) for x in eff_delta)[int(0.95 * (len(eff_delta) - 1))]),
                    },
                },
            },
            layer_B_dynamics={
                "description": "Temporal descriptors of change in geometric state (velocity/acceleration/volatility).",
                "windowing": {"type": "steps", "dt": 1, "smoothing": "ema", "ema_alpha": 0.2},
                "metrics": {
                    "drift_velocity": {"by_layer": drift_velocity_by_layer, "global": drift_velocity_global},
                    "drift_acceleration": {"by_layer": drift_accel_by_layer, "global": drift_accel_global},
                    "volatility": {"by_layer": volatility_by_layer, "global": volatility_global},
                    "alignment_velocity": {"by_layer": alignment_velocity_by_layer, "global": alignment_velocity_global},
                },
                "affected_layers": {"method": "topk_by_abs(drift_velocity)", "k": k, "layers": affected_layers},
            },
            layer_C_event_candidates=None,
            signature=Signature(
                signature_version="sig@2.0",
                construction={
                    "basis_invariant": True,
                    "inputs": ["layer_A_state.metrics", "layer_B_dynamics.metrics"],
                    "includes_layer_C": "optional",
                },
                vector=sig_vec,
                vector_semantics=[
                    "A:eff_rank_delta_mean_abs",
                    "A:sv_entropy_delta_mean_abs",
                    "B:drift_velocity_global",
                    "B:alignment_velocity_global",
                ],
                vector_norm=sig_norm,
            ),
            ranking=Ranking(
                score_version="dis@1.0",
                formula="Magnitude × Coherence × Novelty",
                components=RankingComponents(magnitude=magnitude, coherence=1.0, novelty=1.0),
                total=float(magnitude),
                interpretation="Operational ranking only; not part of squiggle identity or matching.",
            ),
            created_at_utc=_utc_now(),
        )

        out_path = out / f"{family_id}__seed{seed}.json"
        out_path.write_text(ps.model_dump_json(indent=2, by_alias=True), encoding="utf-8")

    return out
