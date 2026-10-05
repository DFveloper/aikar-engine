#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


TRAIN_RE = re.compile(
    r"(?:^|\n)\s*train:.*?data=(?P<current>\d+)/(?P<total>\d+)\s+"
    r"loss(?:_ema(?P<loss_window>\d+))?=(?P<loss>[0-9.eE+-]+).*?"
    r"acc(?:_ema(?P<accuracy_window>\d+))?=(?P<accuracy>[0-9.eE+-]+).*?%\s+"
    r"t=(?P<elapsed>\d+:\d{2}:\d{2})\s+"
    r"ETA=(?P<eta>\d+:\d{2}:\d{2})"
)
CRITICAL_RE = re.compile(
    r"critical_sft:\s+active_tokens=(?P<active>\d+)\s+"
    r"critical_tokens=(?P<critical>\d+).*?"
    r"critical_fraction=(?P<fraction>[0-9.]+).*?"
    r"mean_active_weight=(?P<mean_weight>[0-9.]+).*?"
    r"max_active_weight=(?P<max_weight>[0-9.]+).*?"
    r"unweighted_nll=(?P<nll>[0-9.]+)\s+"
    r"weighted_loss=(?P<weighted_loss>[0-9.]+)"
)
VALIDATION_RE = re.compile(
    r"validation:\s+step=(?P<step>\d+)(?:\s+epoch=(?P<epoch>\d+))?.*?loss=(?P<loss>[0-9.eE+-]+)\s+acc=(?P<accuracy>[0-9.eE+-]+)%"
)
CHECKPOINT_OK_RE = re.compile(r"checkpoint set saved at (?:step|window) (?P<step>\d+)")
CHECKPOINT_FAIL_RE = re.compile(r"checkpoint set failed at (?:step|window) (?P<step>\d+)")
QLORA_CHECKPOINT_PATH_RE = re.compile(r"\.epoch(?P<epoch>\d+)\.ckpt(?P<step>\d+)\.gguf$")
QLORA_CHECKPOINT_TARGET_RE = re.compile(
    r"will save checkpoint every \d+ windows: target=(?P<target>\S+)"
)
SCHEDULE_RE = re.compile(
    r"total_steps=(?P<total_steps>\d+)\s+start_step=(?P<start_step>\d+)"
)
PACKED_TRAIN_RE = re.compile(r"packed windows:\s+train=(?P<windows>\d+)")
EPOCH_RE = re.compile(r"\bepoch\s+(?P<epoch>\d+)\s+lr=")
EPOCH_END_RE = re.compile(r"epoch\s+(?P<epoch>\d+)/\d+:.*?(?:train_loss|adapter:)")
DATASET_RE = re.compile(r"windows\s+\S+\s+(?P<ubatches>\d+)\s+ubatches")
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def duration_seconds(value: str) -> int:
    hours, minutes, seconds = (int(part) for part in value.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def read_tail(path: Path, limit: int = 16 * 1024 * 1024) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        size = path.stat().st_size
        if size <= limit:
            return handle.read().decode("utf-8", errors="replace")
        prefix = handle.read(size - limit).decode("utf-8", errors="replace")
        tail = handle.read().decode("utf-8", errors="replace")
    split = max(prefix.rfind("\n"), prefix.rfind("\r"))
    tail = prefix[split + 1:] + tail
    prefix = prefix[:split + 1]
    context = []
    pending = None
    previous_local = None
    previous_banner = None
    count = 0
    for line in re.split(r"[\r\n]+", prefix):
        match = TRAIN_RE.search(line)
        if match:
            local = int(match["current"])
            if previous_local is not None and local < previous_local and pending:
                context.append(pending)
                pending = None
            pending = line
            previous_local = local
            count += 1
            if count % 100 == 0:
                context.append(pending)
                pending = None
            continue
        banner = EPOCH_RE.search(line)
        if banner and banner["epoch"] == previous_banner:
            continue
        if banner:
            previous_banner = banner["epoch"]
        if banner or EPOCH_END_RE.search(line) or any(marker in line for marker in (
            "packed windows:", "ubatches =", "total_steps=", "validation:",
            "critical_sft:", "checkpoint", "adapter saved to", "critical_stats_every")):
            if pending:
                context.append(pending)
                pending = None
            context.append(line)
    if pending:
        context.append(pending)
    return "\n".join(context) + "\n" + tail


def capture_tmux(session: str) -> str:
    try:
        result = subprocess.run(
            ["tmux", "capture-pane", "-epJt", session, "-S", "-100000"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    return result.stdout if result.returncode == 0 else ""


def training_process(log_path: Path | None = None) -> int | None:
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / "cmdline").read_bytes().decode(errors="ignore").split("\0")
            if not args or Path(args[0]).name not in ("llama-finetune-qlora", "llama-finetune-qlion"):
                continue
            if "--train-file" not in args:
                continue
            if log_path is not None:
                cwd = (entry / "cwd").resolve()
                output = None
                if "--lora-out" in args:
                    output = Path(args[args.index("--lora-out") + 1])
                    if not output.is_absolute():
                        output = cwd / output
                if cwd != log_path.parent and (output is None or output.parent.resolve() != log_path.parent):
                    continue
            return int(entry.name)
        except (OSError, ValueError, IndexError):
            continue
    return None


def downsample(points: list[dict], maximum: int = 240) -> list[dict]:
    if len(points) <= maximum:
        return points
    stride = (len(points) - 1) / (maximum - 1)
    return [points[round(index * stride)] for index in range(maximum)]


def data_steps_per_optimizer_window(local_total: int, windows_per_epoch: int) -> int:
    """Convert the trainer's physical microbatch counter to optimizer windows.

    The final packed window can be short, so ``data_total`` is not always an
    exact multiple of the reported packed-window count.  Rounding the ratio
    preserves the real accumulation factor (for example 43004 / 10757 ≈ 4),
    whereas treating it as one makes train plots four times wider than
    validation plots.
    """
    if local_total <= 0 or windows_per_epoch <= 0:
        return 1
    return max(1, round(local_total / windows_per_epoch))


def ema_series(records: list[dict], window: int) -> list[dict]:
    if not records:
        return []
    alpha = 2.0 / (window + 1.0)
    output: list[dict] = []
    loss_ema: float | None = None
    accuracy_ema: float | None = None
    previous: dict | None = None
    for record in records:
        loss_sample = record["loss"]
        accuracy_sample = record["accuracy"]
        delta = 1
        if previous is not None:
            delta = max(1, record["data_step"] - previous["data_step"])
            if not record["native_ema"] and not previous["native_ema"]:
                # The trainer's loss is cumulative within the current epoch's
                # local `data` counter.  `step` is a WebUI global counter and
                # includes resume offsets, so using it here creates a false,
                # sometimes negative sample after --*-resume.
                local_delta = record["data_step"] - previous["data_step"]
                if local_delta > 0 and record.get("epoch") == previous.get("epoch"):
                    loss_sample = (
                        record["loss"] * record["data_step"]
                        - previous["loss"] * previous["data_step"]
                    ) / local_delta
                else:
                    # A new epoch reset the trainer accumulator. Its first
                    # reported mean is the only valid loss sample available.
                    loss_sample = record["loss"]
                # Legacy accuracy was rounded to 0.01% before logging. Differencing
                # cumulative values amplifies that quantization noise, so smooth the
                # logged value until native per-batch EMA is available.
                accuracy_sample = record["accuracy"]
        if record["native_ema"]:
            loss_ema = record["loss"]
            accuracy_ema = record["accuracy"]
        else:
            effective_alpha = 1.0 - (1.0 - alpha) ** delta
            loss_ema = loss_sample if loss_ema is None else loss_ema + effective_alpha * (loss_sample - loss_ema)
            accuracy_sample = min(100.0, max(0.0, accuracy_sample))
            accuracy_ema = (
                accuracy_sample
                if accuracy_ema is None
                else accuracy_ema + effective_alpha * (accuracy_sample - accuracy_ema)
            )
        output.append({"step": record["step"], "loss": loss_ema, "accuracy": accuracy_ema})
        previous = record
    return output


def ema_dict(records: list[dict], window: int) -> dict | None:
    if not records:
        return None
    alpha = 2.0 / (window + 1.0)
    result = dict(records[0])
    for record in records[1:]:
        for key, sample in record.items():
            result[key] += alpha * (sample - result[key])
    return result


def sparse_ema_series(records: list[dict], key: str, window: int) -> list[dict]:
    if not records:
        return []
    alpha = 2.0 / (window + 1.0)
    result = []
    current: float | None = None
    previous_step: int | None = None
    for record in records:
        step = int(record["step"])
        delta = max(1, step - previous_step) if previous_step is not None else 1
        effective_alpha = 1.0 - (1.0 - alpha) ** delta
        sample = float(record[key])
        current = sample if current is None else current + effective_alpha * (sample - current)
        result.append({"step": step, key: current})
        previous_step = step
    return result


def sparse_metric_ema_series(records: list[dict], window: int) -> list[dict]:
    """EMA for sparse validation measurements, preserving their optimizer-step x-axis."""
    if not records:
        return []
    alpha = 2.0 / (window + 1.0)
    current: dict[str, float] | None = None
    previous_step: int | None = None
    output = []
    for record in records:
        step = int(record["step"])
        delta = max(1, step - previous_step) if previous_step is not None else 1
        effective_alpha = 1.0 - (1.0 - alpha) ** delta
        if current is None:
            current = {"loss": float(record["loss"]), "accuracy": float(record["accuracy"])}
        else:
            for key in current:
                current[key] += effective_alpha * (float(record[key]) - current[key])
        output.append({"step": step, **current})
        previous_step = step
    return output


def token_weighted_ema_series(records: list[dict], key: str, weight_key: str, window: int) -> list[dict]:
    """EMA of a per-token quantity, weighted by supervised tokens per update."""
    if not records:
        return []
    alpha = 2.0 / (window + 1.0)
    numerator: float | None = None
    denominator: float | None = None
    result = []
    for record in records:
        weight = max(0.0, float(record[weight_key]))
        contribution = weight * float(record[key])
        if numerator is None or denominator is None:
            numerator, denominator = contribution, weight
        else:
            numerator = (1.0 - alpha) * numerator + alpha * contribution
            denominator = (1.0 - alpha) * denominator + alpha * weight
        value = numerator / denominator if denominator > 0 else float(record[key])
        result.append({"step": int(record["step"]), key: value})
    return result


def token_weighted_ema(records: list[dict], keys: tuple[str, ...], weight_key: str, window: int) -> dict[str, float]:
    """Return token-weighted EMA values for the requested loss metrics."""
    if not records:
        return {}
    series = {key: token_weighted_ema_series(records, key, weight_key, window) for key in keys}
    return {key: values[-1][key] for key, values in series.items() if values}


def trend_delta(series: list[dict], key: str, window: int) -> dict[str, float | int] | None:
    """Compare the latest EMA value with the value N optimizer windows earlier."""
    if len(series) < 2:
        return None
    latest = series[-1]
    target_step = int(latest["step"]) - window
    baseline = next((point for point in reversed(series[:-1]) if point["step"] <= target_step), None)
    if baseline is None:
        return None
    return {
        "delta": float(latest[key]) - float(baseline[key]),
        "windows": int(latest["step"]) - int(baseline["step"]),
    }


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class Monitor:
    def __init__(self, log_path: Path, tmux_session: str, ema_n: int) -> None:
        self.log_path = log_path
        self.tmux_session = tmux_session
        self.ema_n = ema_n
        self.lock = threading.Lock()
        self.last_step: tuple[int, int] | None = None
        self.last_change = time.monotonic()

    def source(self) -> tuple[str, str]:
        log_text = read_tail(self.log_path)
        if TRAIN_RE.search(log_text):
            return log_text, "log"
        return capture_tmux(self.tmux_session), "tmux"

    def snapshot(self, ema_n: int | None = None) -> dict:
        ema_n = ema_n or self.ema_n
        text, source = self.source()
        text = ANSI_RE.sub("", text).replace("\r", "\n")
        lines = text.splitlines()
        schedule = SCHEDULE_RE.search(text)
        packed = PACKED_TRAIN_RE.search(text)
        dataset = DATASET_RE.search(text)
        start_step = int(schedule["start_step"]) if schedule else 0
        total = int(schedule["total_steps"]) if schedule else 0
        windows = int(packed["windows"]) if packed else 0
        epoch = start_step // windows if windows else 0
        resume_epoch = epoch
        resume_window = start_step % windows if windows else start_step
        epoch_hint = epoch
        previous_local = None
        elapsed_before_epoch = 0
        epoch_elapsed = 0
        records = []
        critical_records = []
        validation_records = []
        successes = set()
        failures = set()
        critical_step = None
        events = []

        for line in lines:
            banner = EPOCH_RE.search(line)
            summary = EPOCH_END_RE.search(line)
            if banner or summary:
                epoch_hint = int((banner or summary)["epoch"])
            match = TRAIN_RE.search(line)
            if match:
                item = match.groupdict()
                local = int(item["current"])
                local_total = int(item["total"])
                next_epoch = max(epoch, epoch_hint)
                if previous_local is not None and local < previous_local:
                    next_epoch = max(next_epoch, epoch + 1)
                if next_epoch > epoch:
                    elapsed_before_epoch += epoch_elapsed
                    epoch_elapsed = 0
                epoch = next_epoch
                factor = int(dataset["ubatches"]) if dataset else data_steps_per_optimizer_window(local_total, windows)
                epoch_windows = windows or max(1, (local_total + factor - 1) // factor)
                offset = resume_window if epoch == resume_epoch else 0
                current = epoch * epoch_windows + offset + local // factor
                epoch_elapsed = duration_seconds(item["elapsed"])
                records.append(dict(step=current + (local % factor) / factor, data_step=local, epoch=epoch,
                                    data_per_window=factor, loss=float(item["loss"]),
                                    accuracy=float(item["accuracy"]),
                                    elapsed_seconds=elapsed_before_epoch + epoch_elapsed,
                                    native_ema=item["loss_window"] is not None))
                previous_local = local
                critical_step = current
                if not total:
                    total = epoch_windows
            critical = CRITICAL_RE.search(line)
            if critical and critical_step is not None:
                values = {key: float(value) for key, value in critical.groupdict().items()}
                critical_records.append(dict(step=critical_step, **values))
            validation = VALIDATION_RE.search(line)
            if validation:
                val_epoch = int(validation["epoch"]) - 1 if validation["epoch"] else epoch
                # Initial resume evaluation and native QAT use global steps.
                val_step = int(validation["step"])
                if records and windows and val_step <= windows:
                    val_step += val_epoch * windows
                validation_records.append(dict(step=val_step,
                                               loss=float(validation["loss"]), accuracy=float(validation["accuracy"])))
            saved = CHECKPOINT_OK_RE.search(line)
            failed = CHECKPOINT_FAIL_RE.search(line)
            if saved:
                successes.add(int(saved["step"]))
            if failed:
                failures.add(int(failed["step"]))
            if "adapter saved to " in line:
                name = line.split("adapter saved to ", 1)[1].strip()
                checkpoint = QLORA_CHECKPOINT_PATH_RE.search(name)
                if checkpoint:
                    successes.add((int(checkpoint["epoch"]) - 1) * windows + int(checkpoint["step"]))
            if any(marker in line for marker in ("checkpoint", "ckpt", "qat_epoch:", "failed to save", "cannot open")):
                events.append(line.strip()[-260:])

        history = ema_series(records, ema_n)
        critical_values = [{key: value for key, value in record.items() if key != "step"} for record in critical_records]
        critical = ema_dict(critical_values, ema_n) or {}
        critical.update(token_weighted_ema(critical_records, ("nll", "weighted_loss"), "active", ema_n))
        nll_history = token_weighted_ema_series(critical_records, "nll", "active", ema_n)
        if not nll_history and not re.search(r"critical_sft: mode=(?:confidence|hybrid|tags)", text):
            nll_history = [{"step": point["step"], "nll": point["loss"]} for point in history]
        validation_records = list({point["step"]: point for point in validation_records}.values())
        validation_history = sparse_metric_ema_series(validation_records, ema_n)
        for target in (() if successes else QLORA_CHECKPOINT_TARGET_RE.finditer(text)):
            path = Path(target["target"])
            for candidate in path.parent.glob(f"{path.name}.epoch*.ckpt*.gguf"):
                checkpoint = QLORA_CHECKPOINT_PATH_RE.search(candidate.name)
                if checkpoint:
                    successes.add((int(checkpoint["epoch"]) - 1) * windows + int(checkpoint["step"]))

        latest = records[-1] if records else None
        current = int(latest["step"]) if latest else start_step
        process_id = training_process(self.log_path)
        activity = (latest["epoch"], latest["data_step"]) if latest else None
        with self.lock:
            if activity is not None and activity != self.last_step:
                self.last_step = activity
                self.last_change = time.monotonic()
            idle_seconds = time.monotonic() - self.last_change
        state = "idle" if process_id is None else "stalled" if idle_seconds > 90 else "running"
        elapsed = latest["elapsed_seconds"] if latest else 0
        rate = (current - start_step) / elapsed if elapsed > 0 else 0.0
        eta = format_duration(max(0, total - current) / rate) if rate > 0 else "--:--:--"
        smoothed = history[-1] if history else {}
        return {
            "state": state, "pid": process_id, "source": source, "updated_at": time.time(),
            "progress": min(1.0, current / total) if total else 0.0,
            "current": current, "total": total, "epoch": epoch + 1,
            "chart_start": start_step, "chart_current": latest["step"] if latest else start_step,
            "loss": smoothed.get("loss"), "accuracy": smoothed.get("accuracy"),
            "unweighted_nll": nll_history[-1]["nll"] if nll_history else None,
            "accuracy_scope": "supervised_token_weighted" if latest and latest["native_ema"] else "legacy_all_positions",
            "ema_n": ema_n, "elapsed": format_duration(elapsed) if latest else "--:--:--",
            "eta": eta, "rate": rate, "history": downsample(history), "nll_history": downsample(nll_history),
            "validation": validation_history[-1] if validation_history else None,
            "validation_history": downsample(validation_history),
            "trends": {"loss": trend_delta(history, "loss", ema_n),
                       "accuracy": trend_delta(history, "accuracy", ema_n),
                       "unweighted_nll": trend_delta(nll_history, "nll", ema_n)},
            "critical": critical,
            "checkpoints": {"last_saved_step": max(successes) if successes else None,
                            "last_failed_step": max(failures) if failures else None,
                            "successful": len(successes), "failed": len(failures)},
            "events": events[-8:][::-1],
        }


class MonitorHandler(BaseHTTPRequestHandler):
    monitor: Monitor
    static_dir: Path

    def log_message(self, format: str, *args: object) -> None:
        return

    def send_bytes(self, payload: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        parsed_url = urlparse(self.path)
        route = parsed_url.path
        if route == "/api/health":
            self.send_bytes(b'{"ok":true}', "application/json; charset=utf-8")
            return
        if route == "/api/status":
            requested_ema = parse_qs(parsed_url.query).get("ema", [None])[0]
            try:
                ema_n = int(requested_ema) if requested_ema is not None else None
            except ValueError:
                ema_n = None
            # Keep the selectable views bounded so an accidental query cannot
            # turn the monitor endpoint into an expensive arbitrary calculation.
            if ema_n not in (25, 50, 100, 250, 500, 1000):
                ema_n = None
            payload = json.dumps(self.monitor.snapshot(ema_n), separators=(",", ":")).encode()
            self.send_bytes(payload, "application/json; charset=utf-8")
            return

        assets = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/index.html": ("index.html", "text/html; charset=utf-8"),
            "/styles.css": ("styles.css", "text/css; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
        }
        asset = assets.get(route)
        if asset is None:
            self.send_bytes(b"Not found", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)
            return
        filename, content_type = asset
        self.send_bytes((self.static_dir / filename).read_bytes(), content_type)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lumen training monitor")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--log", type=Path, default=Path("train.log"))
    parser.add_argument("--tmux-session", default="train")
    parser.add_argument("--ema-n", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    static_dir = Path(__file__).resolve().parent
    if args.ema_n <= 0:
        raise SystemExit("--ema-n must be greater than zero")
    MonitorHandler.monitor = Monitor(args.log.resolve(), args.tmux_session, args.ema_n)
    MonitorHandler.static_dir = static_dir
    server = ThreadingHTTPServer((args.host, args.port), MonitorHandler)
    print(f"Lumen Train Monitor: http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
