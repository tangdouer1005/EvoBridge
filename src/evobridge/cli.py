"""Small CLI for local setup, deterministic demo, and model-backed offline runs."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib
import json
import os
from pathlib import Path
import sys

from .checks import CheckSuite
from .config import RunConfig
from .demo import INITIAL, run_demo
from .engine import Engine
from .harness import Harness
from .policy import PolicyGuard
from .schedules import offline
from .storage import SnapshotStore
from .types import Experience


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description="EvoBridge core framework (no benchmark adapters)")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Run a deterministic fixture without API calls")
    demo.add_argument("--workdir", required=True)
    init = sub.add_parser("init", help="Write an initial M/O/pi harness")
    init.add_argument("--directory", required=True)
    run = sub.add_parser("offline", help="Evolve a fixed local JSONL corpus with configured model endpoints")
    run.add_argument("--workdir", required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--policy", required=True)
    run.add_argument("--harness", required=True)
    run.add_argument("--corpus", required=True)
    run.add_argument("--resources", help="Local JSON mapping of resource names to text")
    run.add_argument("--epochs", type=int, default=5)
    run.add_argument("--local-base-url", default=os.getenv("EVOBRIDGE_LOCAL_BASE_URL", "http://127.0.0.1:8000/v1"))
    run.add_argument("--cloud-base-url", default=os.getenv("EVOBRIDGE_CLOUD_BASE_URL"))
    run.add_argument("--local-key-env", default="EVOBRIDGE_LOCAL_API_KEY")
    run.add_argument("--cloud-key-env", default="EVOBRIDGE_CLOUD_API_KEY")
    run.add_argument("--checks", help="Trusted module:factory returning CheckSuite")
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            print(json.dumps(run_demo(args.workdir), indent=2))
        elif args.command == "init":
            if Path(args.directory).exists():
                raise ValueError("Choose a new directory for the initial harness")
            INITIAL.write_to(args.directory)
            print("Initial memory.md and CONTEXT.md written.")
        elif args.command == "offline":
            from .agents import ModelCloudImprover, ModelLocalImprover
            from .providers import HTTPChatClient
            if not args.cloud_base_url:
                raise ValueError("Set --cloud-base-url or EVOBRIDGE_CLOUD_BASE_URL")
            config = RunConfig.from_dict(load_json(args.config))
            policy = load_json(args.policy)
            if set(policy) != {"prohibited_patterns"}:
                raise ValueError("Policy must contain prohibited_patterns only")
            guard = PolicyGuard(policy["prohibited_patterns"])
            corpus = [Experience.from_dict(json.loads(line)) for line in
                      Path(args.corpus).read_text(encoding="utf-8").splitlines() if line.strip()]
            checks = CheckSuite()
            if args.checks:
                module, sep, factory = args.checks.partition(":")
                if not sep:
                    raise ValueError("Checks factory must be module:callable")
                checks = getattr(importlib.import_module(module), factory)()
                if not isinstance(checks, CheckSuite):
                    raise ValueError("Checks factory must return CheckSuite")
            local = HTTPChatClient(args.local_base_url, config.local,
                                   api_key_env=args.local_key_env if os.getenv(args.local_key_env) else None)
            cloud = HTTPChatClient(args.cloud_base_url, config.cloud, api_key_env=args.cloud_key_env)
            store = SnapshotStore(args.workdir)
            engine = Engine(store, ModelLocalImprover(local, checks=checks), ModelCloudImprover(cloud),
                            guard, checks, config, initial=Harness.from_directory(args.harness),
                            resources=load_json(args.resources) if args.resources else {})
            try:
                snapshots = offline(engine, corpus, args.epochs)
            finally:
                for role, client in (("local", local), ("cloud", cloud)):
                    for receipt in client.usage_receipts:
                        data = asdict(receipt)
                        store.save_event("provider_usage", {"role": role, **data})
            rounds = store.history("rounds")
            failed = sum(row["result"]["status"] == "failed" for row in rounds)
            print(json.dumps({"checkpoints": {str(e): h.id for e, h in snapshots.items()},
                              "failed_rounds": failed}, indent=2))
            return 1 if failed else 0
    except Exception as exc:
        # Provider exception messages can contain private content; do not echo
        # them into shared terminal logs. The local store retains diagnostics.
        print(f"EvoBridge stopped ({type(exc).__name__}). Check local configuration and private run records.", file=sys.stderr)
        return 1
    return 0
