import argparse

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

    args = ap.parse_args()

    families = [x.strip() for x in str(args.families).split(",") if x.strip()]
    layers = [int(x.strip()) for x in str(args.layers).split(",") if x.strip()]

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
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
