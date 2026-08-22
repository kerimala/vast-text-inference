#!/usr/bin/env python3
"""Cost-gated Vast.ai search and deployment helper for pinned model profiles."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "aeon-bf16-a100.json"
DEFAULT_SSH_KEY = Path.home() / ".ssh" / "vast_ai_ed25519"
DEFAULT_HF_TOKEN_FILE = Path.home() / ".cache" / "huggingface" / "token"
TERMINAL_FAILURE_STATES = {"error", "exited", "offline"}


class LabError(RuntimeError):
    pass


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def verify_hf_access(config: dict[str, Any], token: str) -> None:
    model_id = config["model_id"]
    revision = config["model_revision"]
    url = f"https://huggingface.co/{model_id}/resolve/{revision}/config.json"
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}"},
        method="HEAD",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            if response.status != 200:
                raise LabError(f"Hugging Face access check returned HTTP {response.status}.")
    except urllib.error.HTTPError as exc:
        raise LabError(f"Hugging Face access check returned HTTP {exc.code}.") from exc
    except urllib.error.URLError as exc:
        raise LabError(f"Hugging Face access check failed: {exc.reason}") from exc


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, text=True, capture_output=True)


def vast_json(arguments: list[str], *, sensitive_values: tuple[str, ...] = ()) -> Any:
    result = run(["vastai", *arguments, "--raw"], check=False)
    if result.returncode != 0:
        diagnostic = (result.stderr or result.stdout or "unknown error").strip()[:500]
        for value in sensitive_values:
            if value:
                diagnostic = diagnostic.replace(value, "<redacted>")
        raise LabError(f"Vast CLI failed: {diagnostic}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        diagnostic = result.stdout[:300]
        for value in sensitive_values:
            if value:
                diagnostic = diagnostic.replace(value, "<redacted>")
        raise LabError(f"Vast CLI returned invalid JSON: {diagnostic}") from exc


def offer_query(config: dict[str, Any]) -> str:
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
        f"dph<={config['max_hourly_usd']}",
    ]
    if rules["verified_required"]:
        comparisons.append("verified=true")
    if rules.get("compute_cap_min") is not None:
        comparisons.append(f"compute_cap>={rules['compute_cap_min']}")
    return " ".join(comparisons)


def fetch_offers(config: dict[str, Any], *, machine_id: int | None = None, limit: int = 50) -> list[dict[str, Any]]:
    payload = vast_json(
        [
            "search",
            "offers",
            offer_query(config),
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
    valid = [offer for offer in payload if validate_offer(config, offer) == []]
    if machine_id is not None:
        valid = [offer for offer in valid if int(offer.get("machine_id", -1)) == machine_id]
    return valid


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
    if float(offer.get("compute_cap") or 0) < float(rules.get("compute_cap_min") or 0):
        failures.append("GPU compute capability is too low")
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
    is_verified = (
        bool(offer.get("verified"))
        or offer.get("verification") == "verified"
        or offer.get("vericode") == 1
    )
    if rules["verified_required"] and not is_verified:
        failures.append("Host is not verified")
    return failures


def public_offer(offer: dict[str, Any]) -> dict[str, Any]:
    is_verified = (
        bool(offer.get("verified"))
        or offer.get("verification") == "verified"
        or offer.get("vericode") == 1
    )
    return {
        "offer_id": offer.get("id"),
        "machine_id": offer.get("machine_id"),
        "gpu": offer.get("gpu_name"),
        "gpu_ram_mb": offer.get("gpu_ram"),
        "hourly_usd_with_storage": round(hourly_price(offer), 4),
        "reliability": offer.get("reliability"),
        "verified": is_verified,
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
    command = [
        "systemd-run",
        "--user",
        f"--unit={unit}",
        f"--on-active={ttl_minutes}m",
    ]
    # Keep account selection consistent for the delayed destroy command when
    # deployment uses an isolated Vast CLI configuration (for example, the
    # business account alongside an older private account).
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config_home:
        command.append(f"--setenv=XDG_CONFIG_HOME={xdg_config_home}")
    command.extend(
        [
            vastai_path,
            "destroy",
            "instance",
            str(instance_id),
            "--yes",
        ]
    )
    result = run(command, check=False)
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
                    "if test -f /workspace/vast-text-inference/state/failed; then exit 42; fi; "
                    "test -f /workspace/vast-text-inference/state/ready",
                ],
                check=False,
            )
            if probe.returncode == 42:
                raise LabError("Remote provisioning reported failure.")
            if probe.returncode == 0:
                return ssh_url
        time.sleep(15)
    raise LabError(f"Startup timeout after {timeout_minutes} minutes")


def command_search(config: dict[str, Any], limit: int) -> int:
    offers = fetch_offers(config, limit=limit)
    print(json.dumps([public_offer(offer) for offer in offers], indent=2))
    return 0 if offers else 2


def selected_offer(config: dict[str, Any], machine_id: int) -> dict[str, Any]:
    # Offer IDs can rotate between searches. Select by stable machine ID, then
    # use the fresh offer ID returned by the final pre-rental search.
    offers = fetch_offers(config, machine_id=machine_id, limit=200)
    if not offers:
        raise LabError("The selected machine vanished or no longer satisfies the profile.")
    return min(offers, key=hourly_price)


def cost_quote(config: dict[str, Any], offer: dict[str, Any], ttl_minutes: int) -> dict[str, Any]:
    rental_cost = hourly_price(offer) * ttl_minutes / 60
    download_cost = float(offer.get("inet_down_cost") or 0) * config["estimated_first_download_gb"]
    estimated_total = rental_cost + download_cost
    return {
        "ttl_minutes": ttl_minutes,
        "estimated_rental_cost_usd": round(rental_cost, 2),
        "estimated_first_model_download_cost_usd": round(download_cost, 2),
        "estimated_total_cost_usd": round(estimated_total, 2),
        "required_confirmation": f"RENT MACHINE {offer['machine_id']} UP TO USD {estimated_total:.2f}",
    }


def command_quote(config: dict[str, Any], args: argparse.Namespace) -> int:
    if args.ttl_minutes < 30 or args.ttl_minutes > 240:
        raise LabError("TTL must be between 30 and 240 minutes.")
    offer = selected_offer(config, args.machine_id)
    quote = cost_quote(config, offer, args.ttl_minutes)
    print(
        json.dumps(
            {
                "offer": public_offer(offer),
                **quote,
            },
            indent=2,
        )
    )
    return 0


def command_deploy(config: dict[str, Any], args: argparse.Namespace) -> int:
    if not args.execute:
        raise LabError("Refusing paid action without --execute.")
    if args.ttl_minutes < 30 or args.ttl_minutes > 240:
        raise LabError("TTL must be between 30 and 240 minutes.")
    if not DEFAULT_SSH_KEY.is_file():
        raise LabError(f"Dedicated SSH key is missing: {DEFAULT_SSH_KEY}")
    if not config.get("template_hash"):
        raise LabError("No Vast template hash is recorded in the profile yet.")

    hf_token = ""
    if config.get("requires_hf_token"):
        token_file = args.hf_token_file.expanduser()
        if not token_file.is_file():
            raise LabError(f"Hugging Face token file is missing: {token_file}")
        hf_token = token_file.read_text(encoding="utf-8").strip()
        if not hf_token:
            raise LabError(f"Hugging Face token file is empty: {token_file}")
        verify_hf_access(config, hf_token)

    if config.get("runtime", {}).get("openwebui_port") is not None:
        if not args.webui_public_url:
            raise LabError("--webui-public-url is required for an Open WebUI profile.")
        if not args.webui_public_url.startswith("https://"):
            raise LabError("--webui-public-url must use HTTPS.")

    offer = selected_offer(config, args.machine_id)
    quote = cost_quote(config, offer, args.ttl_minutes)
    if args.confirm != quote["required_confirmation"]:
        raise LabError(f"Cost confirmation must exactly equal: {quote['required_confirmation']}")

    print(json.dumps(public_offer(offer), indent=2))
    print(json.dumps(quote, indent=2))
    print("Local TTL guard will be armed immediately after creation.")
    create_arguments = [
        "create",
        "instance",
        str(offer["id"]),
        "--template_hash",
        str(config["template_hash"]),
        "--disk",
        str(config["disk_gb"]),
        "--label",
        config["profile"],
        "--cancel-unavail",
    ]
    if hf_token:
        docker_options = str(config.get("docker_options") or "").strip()
        docker_options = f"{docker_options} -e HF_TOKEN={hf_token}".strip()
        if args.webui_public_url:
            docker_options = f"{docker_options} -e WEBUI_PUBLIC_URL={args.webui_public_url}"
        create_arguments.extend(["--env", docker_options])
    created = vast_json(create_arguments, sensitive_values=(hf_token,))
    hf_token = ""
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
    runtime = config.get("runtime", {})
    vllm_port = int(runtime.get("vllm_port", 8000))
    openwebui_port = runtime.get("openwebui_port")
    forwards = [f"-L {vllm_port}:127.0.0.1:{vllm_port}"]
    if openwebui_port is not None:
        openwebui_port = int(openwebui_port)
        forwards.append(f"-L {openwebui_port}:127.0.0.1:{openwebui_port}")
    print("Ready. Keep this tunnel running:")
    print(f"ssh -N -i {DEFAULT_SSH_KEY} -p {port} {' '.join(forwards)} {host_part}")
    if openwebui_port is not None:
        print(f"Open WebUI is tunneled to http://127.0.0.1:{openwebui_port}")
    print("Then run: python3 scripts/smoke_test.py")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser("search", help="List matching offers without renting")
    search_parser.add_argument("--limit", type=int, default=20)

    quote_parser = subparsers.add_parser("quote", help="Revalidate one offer and print its confirmation text")
    quote_parser.add_argument("machine_id", type=int)
    quote_parser.add_argument("--ttl-minutes", type=int, default=120)

    deploy_parser = subparsers.add_parser("deploy", help="Rent one revalidated offer with a hard TTL")
    deploy_parser.add_argument("machine_id", type=int)
    deploy_parser.add_argument("--ttl-minutes", type=int, default=120)
    deploy_parser.add_argument("--confirm", required=True)
    deploy_parser.add_argument("--hf-token-file", type=Path, default=DEFAULT_HF_TOKEN_FILE)
    deploy_parser.add_argument("--webui-public-url")
    deploy_parser.add_argument("--execute", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    try:
        if args.command == "search":
            return command_search(config, args.limit)
        if args.command == "quote":
            return command_quote(config, args)
        return command_deploy(config, args)
    except (LabError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
