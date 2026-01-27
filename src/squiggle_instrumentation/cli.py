import argparse
from pathlib import Path

from .probe_runner import run_probe_experiment


def main() -> int:
    ap = argparse.ArgumentParser(prog="squiggle-probe")
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--base-checkpoint", required=True)
    ap.add_argument("--families", required=True, help="Comma-separated family_id list")

    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="fp32")

    ap.add_argument("--probe-name", default="layerwise_geometry_v1")
    ap.add_argument("--layers", default="0,4,8,12")
    ap.add_argument("--lr", type=float, default=1e-4)

    ap.add_argument(
        "--summaries-dir",
        default=None,
        help="Directory to write ProbeSummary JSONs (defaults under SQUIGGLE_DATA_ROOT)",
    )

    # HuggingFace adapter options
    ap.add_argument(
        "--adapter",
        choices=["dummy", "hf"],
        default="dummy",
        help="Adapter type: 'dummy' for testing, 'hf' for HuggingFace models",
    )
    ap.add_argument(
        "--hf-model",
        default=None,
        help="HuggingFace model name or path (required when --adapter=hf)",
    )
    ap.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRA rank (default: 8)",
    )
    ap.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha (default: 16)",
    )
    ap.add_argument(
        "--family-data-dir",
        default=None,
        help="Directory containing family JSONL files for training data",
    )

    args = ap.parse_args()

    families = [x.strip() for x in str(args.families).split(",") if x.strip()]
    layers = [int(x.strip()) for x in str(args.layers).split(",") if x.strip()]

    # Create adapter
    adapter = None
    family_data = {}

    if args.adapter == "hf":
        if not args.hf_model:
            ap.error("--hf-model is required when --adapter=hf")

        try:
            from .hf_adapter import HFProbeAdapter, load_family_training_data
        except ImportError as e:
            ap.error(
                f"HF adapter requires extra dependencies. "
                f"Install with: pip install squiggle-instrumentation[hf]\n"
                f"Error: {e}"
            )

        # Determine torch dtype
        import torch

        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(args.dtype, torch.float32)

        adapter = HFProbeAdapter(
            model_name_or_path=args.hf_model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            torch_dtype=torch_dtype,
        )

        # Load family training data if provided
        if args.family_data_dir:
            data_dir = Path(args.family_data_dir)
            for fam_id in families:
                fam_file = data_dir / f"{fam_id}.jsonl"
                if fam_file.exists():
                    family_data[fam_id] = load_family_training_data(fam_file)
                    print(f"Loaded {len(family_data[fam_id])} training items for {fam_id}")

    run_probe_experiment(
        exp_id=str(args.exp_id),
        model_id=str(args.model_id),
        base_checkpoint=str(args.base_checkpoint),
        family_ids=families,
        steps=int(args.steps),
        lr=float(args.lr),
        seed=int(args.seed),
        device=str(args.device),
        dtype=str(args.dtype),
        probe_name=str(args.probe_name),
        layers_covered=layers,
        summaries_dir=(str(args.summaries_dir) if args.summaries_dir else None),
        adapter=adapter,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
