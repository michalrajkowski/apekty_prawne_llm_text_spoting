"""Top-level CLI entrypoint for APM workflows."""

from __future__ import annotations

import argparse

from apm.experiments import runner as experiment_runner


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="APM command-line tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_experiment_parser = subparsers.add_parser(
        "run-experiment",
        help="Run immutable detector scoring experiment.",
    )
    run_experiment_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to `apm.experiments.runner`.",
    )
    return parser


def main() -> int:
    """Dispatch CLI subcommands."""

    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "run-experiment":
        forwarded = list(args.args)
        run_parser = experiment_runner.build_arg_parser()
        run_args = run_parser.parse_args(forwarded)
        request = experiment_runner.build_request_from_args(run_args)
        result = experiment_runner.run_experiment(request)
        print(result.run_dir)
        return 0

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
