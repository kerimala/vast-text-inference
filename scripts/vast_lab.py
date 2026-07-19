#!/usr/bin/env python3
"""Cost-gated Vast.ai search and deployment helper for the AEON BF16 baseline."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "aeon-bf16-a100.json"
DEFAULT_SSH_KEY = Path.home() / ".ssh" / "vast_ai_ed25519"
TERMINAL_FAILURE_STATES = {"error", "exited", "offline"}


class LabError(RuntimeError):
    pass


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, text=True, capture_output=True)


def vast_json(arguments: list[str]) -> Any:
    result = run(["vastai", *arguments, "--raw"])
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise LabError(f"Vast CLI returned invalid JSON: {result.stdout[:300]}") from exc


def offer_query(config: dict[str, Any], offer_id: int | None = None) -> str:
    rules = config["offer"]
    comparisons = [
        "rentable=true",
        f"num_gpus={rules['num_gpus']}",
        f"gpu_ram>={rules['gpu_ram_min_mb'] / 1024:g}",
        f"gpu_ram<={rules['gpu_ram_max_mb'] / 1024:g}",
        f"cpu_arch={rules['cpu_arch']}",
        f"reliability>={rules['reliability_min']}",
        f"inet_down>={rules['inet_down_min_mbps']}",
        f"direct_port_count>={rules['direct_port_count_min']}",
        f"disk_space>={config['disk_gb']}",
    ]
    if rules["verified_required"]:
        comparisons.append("verified=true")
    if offer_id is not None:
        comparisons.append(f"id={offer_id}")
    return " ".join(comparisons)


def fetch_offers(config: dict[str, Any], *, offer_id: int | None = None, limit: int = 50) -> list[dict[str, Any]]:
    payload = vast_json(
        [
            "search",
            "offers",
            offer_query(config, offer_id),
            "--no-default",
            "--storage",
            str(config["disk_gb"]),
            "--order",
            "dph",
            "--limit",
            str(limit),
        ]
    )
    if not isinstance(payload, list):
        raise LabError("Unexpected Vast offer response.")
    return [offer for offer in payload if validate_offer(config, offer) == []]


def hourly_price(offer: dict[str, Any]) -> float:
    return float(offer.get("dph_total") or offer.get("dph") or 0)


def validate_offer(config: dict[str, Any], offer: dict[str, Any]) -> list[str]:
    rules = config["offer"]
    failures: list[str] = []
    if offer.get("gpu_name") not in rules["gpu_names"]:
        failures.append("GPU model is outside the allowlist")
    if int(offer.get("num_gpus") or 0) != rules["num_gpus"]:
        failures.append("GPU count differs")
    gpu_ram = float(offer.get("gpu_ram") or 0)
    if not rules["gpu_ram_min_mb"] <= gpu_ram <= rules["gpu_ram_max_mb"]:
        failures.append("GPU RAM differs")
    if offer.get("cpu_arch") != rules["cpu_arch"]:
        failures.append("CPU architecture differs")
    if float(offer.get("reliability") or 0) < rules["reliability_min"]:
        failures.append("Reliability is too low")
    if float(offer.get("inet_down") or 0) < rules["inet_down_min_mbps"]:
        failures.append("Download bandwidth is too low")
    if int(offer.get("direct_port_count") or 0) < rules["direct_port_count_min"]:
        failures.append("No direct SSH port is available")
    if float(offer.get("disk_space") or 0) < config["disk_gb"]:
        failures.append("Not enough disk")
    if hourly_price(offer) > config["max_hourly_usd"]:
        failures.append("Hourly price exceeds the cap")
    if rules["verified_required"] and not offer.get("verified"):
        failures.append("Host is not verified")
    return failures


def public_offer(offer: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": offer.get("id"),
        "gpu": offer.get("gpu_name"),
        "gpu_ram_mb": offer.get("gpu_ram"),
        "hourly_usd_with_storage": round(hourly_price(offer), 4),
        "reliability": offer.get("reliability"),
        "verified": offer.get("verified"),
        "inet_down_mbps": offer.get("inet_down"),
        "direct_port_count": offer.get("direct_port_count"),
        "disk_space_gb": offer.get("disk_space"),
        "location": offer.get("geolocation"),
        "cpu_arch": offer.get("cpu_arch"),
    }


def destroy_instance(instance_id: int) -> None:
    result = run(["vastai", "destroy", "instance", str(instance_id), "--yes"], check=False)
    if result.returncode != 0:
        raise LabError(f"Could not destroy instance {instance_id}: {result.stderr.strip()}")


def arm_cleanup(instance_id: int, ttl_minutes: int) -> str:
    unit = f"vast-lab-destroy-{instance_id}"
    vastai_path = shutil.which("vastai")
    if not vastai_path:
        raise LabError("vastai is not available in PATH.")
    result = run(
        [
            "systemd-run",
            "--user",
            f"--unit={unit}",
            f"--on-active={ttl_minutes}m",
            vastai_path,
            "destroy",
            "instance",
            str(instance_id),
            "--yes",
        ],
        check=False,
    )
    if result.returncode != 0:
        raise LabError(f"Could not arm cleanup timer: {result.stderr.strip()}")
    return unit


def instance_rows() -> list[dict[str, Any]]:
    payload = vast_json(["show", "instances"])
    if not isinstance(payload, list):
        raise LabError("Unexpected Vast instance response.")
    return payload


def wait_for_ready(instance_id: int, timeout_minutes: int, ssh_key: Path) -> str:
    deadline = time.monotonic() + timeout_minutes * 60
    ssh_url = ""
    while time.monotonic() < deadline:
        row = next((item for item in instance_rows() if int(item.get("id", -1)) == instance_id), None)
        if row is None:
            raise LabError(f"Instance {instance_id} disappeared from the account.")
        status = str(row.get("actual_status") or row.get("cur_state") or "unknown").lower()
        print(f"Instance {instance_id}: {status}", flush=True)
        if status in TERMINAL_FAILURE_STATES:
            raise LabError(f"Instance entered terminal state: {status}")

        url_result = run(["vastai", "ssh-url", str(instance_id)], check=False)
        candidate = url_result.stdout.strip()
        if url_result.returncode == 0 and re.fullmatch(r"ssh://[^\s]+", candidate):
            ssh_url = candidate
            target = candidate.removeprefix("ssh://")
            host_part, _, port = target.rpartition(":")
            probe = run(
                [
                    "ssh",
                    "-i",
                    str(ssh_key),
                    "-p",
                    port,
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=8",
                    "-o",
                    "StrictHostKeyChecking=accept-new",
                    host_part,
                    "test -f /workspace/vast-text-inference/state/ready",
                ],
                check=False,
            )
            if probe.returncode == 0:
                return ssh_url
        time.sleep(15)
    raise LabError(f"Startup timeout after {timeout_minutes} minutes")


def command_search(config: dict[str, Any], limit: int) -> int:
    offers = fetch_offers(config, limit=limit)
    print(json.dumps([public_offer(offer) for offer in offers], indent=2))
    return 0 if offers else 2


def command_deploy(config: dict[str, Any], args: argparse.Namespace) -> int:
    if not args.execute:
        raise LabError("Refusing paid action without --execute.")
    if args.ttl_minutes < 30 or args.ttl_minutes > 240:
        raise LabError("TTL must be between 30 and 240 minutes.")
    if not DEFAULT_SSH_KEY.is_file():
        raise LabError(f"Dedicated SSH key is missing: {DEFAULT_SSH_KEY}")
    if not config.get("template_hash"):
        raise LabError("No Vast template hash is recorded in the profile yet.")

    offers = fetch_offers(config, offer_id=args.offer_id, limit=5)
    if len(offers) != 1:
        raise LabError("The selected offer vanished or no longer satisfies the profile.")
    offer = offers[0]
    price = hourly_price(offer)
    max_cost = price * args.ttl_minutes / 60
    expected = f"RENT {args.offer_id} UP TO ${max_cost:.2f}"
    if args.confirm != expected:
        raise LabError(f"Cost confirmation must exactly equal: {expected}")

    print(json.dumps(public_offer(offer), indent=2))
    print(f"Hard TTL: {args.ttl_minutes} minutes; estimated upper rental cost: ${max_cost:.2f}")
    created = vast_json(
        [
            "create",
            "instance",
            str(args.offer_id),
            "--template_hash",
            str(config["template_hash"]),
            "--disk",
            str(config["disk_gb"]),
            "--label",
            config["profile"],
            "--cancel-unavail",
        ]
    )
    instance_id = int(created.get("new_contract") or created.get("id") or 0)
    if not instance_id:
        raise LabError(f"Vast did not return an instance ID: {created}")

    try:
        timer_unit = arm_cleanup(instance_id, args.ttl_minutes)
    except Exception:
        destroy_instance(instance_id)
        raise

    print(f"Instance {instance_id} created; cleanup timer {timer_unit} is armed.")
    try:
        ssh_url = wait_for_ready(instance_id, config["runtime"]["startup_timeout_minutes"], DEFAULT_SSH_KEY)
    except Exception:
        destroy_instance(instance_id)
        raise

    target = ssh_url.removeprefix("ssh://")
    host_part, _, port = target.rpartition(":")
    print("Ready. Keep this tunnel running:")
    print(f"ssh -N -i {DEFAULT_SSH_KEY} -p {port} -L 8000:127.0.0.1:8000 {host_part}")
    print("Then run: python3 scripts/smoke_test.py")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser("search", help="List matching offers without renting")
    search_parser.add_argument("--limit", type=int, default=20)

    deploy_parser = subparsers.add_parser("deploy", help="Rent one revalidated offer with a hard TTL")
    deploy_parser.add_argument("offer_id", type=int)
    deploy_parser.add_argument("--ttl-minutes", type=int, default=120)
    deploy_parser.add_argument("--confirm", required=True)
    deploy_parser.add_argument("--execute", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    try:
        if args.command == "search":
            return command_search(config, args.limit)
        return command_deploy(config, args)
    except (LabError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
