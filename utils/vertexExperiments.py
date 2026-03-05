from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Callable, Dict, Optional
import torch
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional runtime dependency
    SummaryWriter = None


class VertexExperiments:
    """Vertex experiment/event tracking wrapper with safe fallbacks."""

    def __init__(
        self,
        *,
        enabled: bool,
        is_main_process: bool,
        event_path: Optional[str],
        tensorboard_dir: Optional[str],
        enable_tensorboard: bool,
        tensorboard_hist_interval_steps: int,
        run_config: Dict[str, Any],
        log_interval_steps: int,
        logger: Optional[Callable[[str], None]] = None,
        log_exception: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.enabled = bool(enabled and is_main_process)
        self.event_path = event_path
        self.run_config = dict(run_config or {})
        self.log_interval_steps = int(log_interval_steps or 0)
        self.tensorboard_dir = tensorboard_dir
        self.enable_tensorboard = bool(enable_tensorboard and is_main_process)
        self.tensorboard_hist_interval_steps = int(tensorboard_hist_interval_steps or 0)
        self.tensorboard_writer = None
        self._log = logger
        self._log_exception = log_exception
        self._vertex_sdk = None
        self._vertex_run_started = False
        self._vertex_tracking_finalized = False
        self.experiment_name: Optional[str] = None
        self.run_name: Optional[str] = None

    def _safe_log(self, message: str) -> None:
        if self._log is None:
            return
        try:
            self._log(message)
        except Exception:
            pass

    def _safe_log_exception(self, prefix: str, exc: Optional[Exception] = None) -> None:
        message = prefix if exc is None else f"{prefix}: {exc}"
        if self._log_exception is not None:
            try:
                self._log_exception(message)
                return
            except Exception:
                pass
        self._safe_log(message)

    def _init_tensorboard(self) -> None:
        if not self.enable_tensorboard:
            return
        if not self.tensorboard_dir:
            return
        if SummaryWriter is None:
            self._safe_log("[Trainer] Warning: TensorBoard writer unavailable (tensorboard not installed).")
            return
        try:
            os.makedirs(self.tensorboard_dir, exist_ok=True)
            self.tensorboard_writer = SummaryWriter(log_dir=self.tensorboard_dir)
            self.track_event("tensorboard_init", {"dir": self.tensorboard_dir}, global_step=0)
        except Exception as exc:
            self._safe_log_exception("[Trainer] Warning: Failed to initialize TensorBoard writer", exc)
            self.tensorboard_writer = None

    def _close_tensorboard(self) -> None:
        if self.tensorboard_writer is None:
            return
        try:
            self.tensorboard_writer.flush()
            self.tensorboard_writer.close()
        except Exception as exc:
            self._safe_log_exception("[Trainer] Warning: Failed to close TensorBoard writer", exc)
        finally:
            self.tensorboard_writer = None

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): VertexExperiments._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [VertexExperiments._json_safe(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.utcnow().isoformat() + "Z"

    @staticmethod
    def _get_env(*names: str) -> Optional[str]:
        for name in names:
            value = os.environ.get(name)
            if value:
                return value
        return None

    def _append_jsonl(self, payload: Dict[str, Any]) -> None:
        if not self.enabled or not self.event_path:
            return
        try:
            os.makedirs(os.path.dirname(self.event_path), exist_ok=True)
            with open(self.event_path, "a", encoding="utf-8") as fp:
                fp.write(json.dumps(payload, ensure_ascii=True) + "\n")
        except Exception:
            self._safe_log_exception("[Trainer] Warning: Failed to append Vertex event")

    def track_event(self, event: str, payload: Optional[Dict[str, Any]], global_step: int) -> None:
        if not self.enabled:
            return
        entry = {
            "ts": self._utc_now_iso(),
            "event": str(event),
            "global_step": int(global_step),
            "run_name": self.run_name,
            "experiment": self.experiment_name,
            "payload": self._json_safe(payload or {}),
        }
        self._append_jsonl(entry)

    def start(self, *, global_step: int, device: str, world_size: int, enable_metrics: bool) -> None:
        self._init_tensorboard()
        if not self.enabled:
            return
        self.track_event(
            "run_start",
            {
                "device": device,
                "world_size": int(world_size),
                "enable_metrics": bool(enable_metrics),
                "config": self.run_config,
            },
            global_step=global_step,
        )

        try:
            from google.cloud import aiplatform
        except Exception:
            self.track_event("vertex_sdk_unavailable", {}, global_step=global_step)
            return

        project = self._get_env("TRAIN_VERTEX_PROJECT", "GOOGLE_CLOUD_PROJECT", "CLOUD_ML_PROJECT_ID")
        location = self._get_env("TRAIN_VERTEX_LOCATION", "AIP_REGION", "CLOUD_ML_REGION", "VERTEX_REGION")
        self.experiment_name = self._get_env(
            "TRAIN_VERTEX_EXPERIMENT_NAME",
            "AIP_EXPERIMENT_NAME",
        ) or "distilled-llm-train"
        job_name = self._get_env("AIP_JOB_NAME", "CLOUD_ML_JOB_ID", "AIP_CUSTOM_JOB_ID") or "local"
        rank = self._get_env("RANK") or "0"
        self.run_name = self._get_env("TRAIN_VERTEX_RUN_NAME") or f"{job_name}-r{rank}"

        init_kwargs: Dict[str, Any] = {"experiment": self.experiment_name}
        if project:
            init_kwargs["project"] = str(project)
        if location:
            init_kwargs["location"] = str(location)

        try:
            aiplatform.init(**init_kwargs)
            try:
                # Ensure experiment metadata context exists before creating a run.
                aiplatform.Experiment.get_or_create(
                    experiment_name=self.experiment_name,
                )
            except TypeError:
                # Backward compatibility with SDK versions that differ on kwargs.
                aiplatform.Experiment.get_or_create(self.experiment_name)

            try:
                # First attempt should create a new run context.
                aiplatform.start_run(run=self.run_name, resume=False)
            except TypeError:
                aiplatform.start_run(self.run_name, resume=False)
            except Exception as first_start_exc:
                # If run already exists, fallback to resume mode.
                msg = str(first_start_exc).lower()
                if "already exists" in msg or "alreadyexists" in msg:
                    try:
                        aiplatform.start_run(run=self.run_name, resume=True)
                    except TypeError:
                        aiplatform.start_run(self.run_name, resume=True)
                else:
                    raise
            self._vertex_sdk = aiplatform
            self._vertex_run_started = True

            run_params = {
                "n_layer": self.run_config.get("n_layer"),
                "n_embd": self.run_config.get("n_embd"),
                "n_head": self.run_config.get("n_head"),
                "block_size": self.run_config.get("block_size"),
                "optimizer": self.run_config.get("optimizer"),
                "lr": self.run_config.get("lr"),
                "weight_decay": self.run_config.get("weight_decay"),
                "train_batch_size": self.run_config.get("train_batch_size"),
                "grad_accum_steps": self.run_config.get("grad_accum_steps"),
                "total_steps": self.run_config.get("total_steps"),
                "warmup_steps": self.run_config.get("warmup_steps"),
            }
            run_params = {k: v for k, v in run_params.items() if v is not None}
            if run_params:
                try:
                    aiplatform.log_params(run_params)
                except Exception:
                    self._safe_log_exception("[Trainer] Warning: Failed to log Vertex params")

            self.track_event(
                "vertex_sdk_started",
                {
                    "project": project,
                    "location": location,
                    "experiment": self.experiment_name,
                    "run_name": self.run_name,
                },
                global_step=global_step,
            )
        except Exception as exc:
            self.track_event(
                "vertex_sdk_init_failed",
                {"error": str(exc), "project": project, "location": location},
                global_step=global_step,
            )
            self._safe_log_exception("[Trainer] Warning: Failed to initialize Vertex SDK tracking", exc)
            self._vertex_sdk = None
            self._vertex_run_started = False

    def log_metrics(self, *, metrics: Dict[str, float], step: int, force: bool = False) -> None:
        if not self.enabled:
            return
        if not force and self.log_interval_steps > 0 and int(step) % self.log_interval_steps != 0:
            return
        if not self._vertex_run_started or self._vertex_sdk is None:
            return
        try:
            self._vertex_sdk.log_metrics(metrics=metrics, step=int(step))
        except TypeError:
            self._vertex_sdk.log_metrics(metrics)
        except Exception as exc:
            self._safe_log_exception("[Trainer] Warning: Failed to log Vertex metrics", exc)

    def log_tensorboard_step(
        self,
        *,
        step: int,
        train_loss: float,
        lr: float,
        grad_norm: Optional[float],
        val_loss: Optional[float],
        iter_time_s: Optional[float],
        model: Any,
        memory_summary: Dict[str, Any],
    ) -> None:
        if self.tensorboard_writer is None:
            return
        try:
            self.tensorboard_writer.add_scalar("train/loss", float(train_loss), int(step))
            self.tensorboard_writer.add_scalar("train/lr", float(lr), int(step))
            if grad_norm is not None:
                self.tensorboard_writer.add_scalar("train/grad_norm", float(grad_norm), int(step))
            if val_loss is not None:
                self.tensorboard_writer.add_scalar("val/loss", float(val_loss), int(step))
            if iter_time_s is not None:
                self.tensorboard_writer.add_scalar("perf/iter_time_ms", float(iter_time_s) * 1000.0, int(step))
            if memory_summary:
                if "allocated_gb" in memory_summary:
                    self.tensorboard_writer.add_scalar(
                        "system/gpu_mem_allocated_gb",
                        float(memory_summary["allocated_gb"]),
                        int(step),
                    )
                if "reserved_gb" in memory_summary:
                    self.tensorboard_writer.add_scalar(
                        "system/gpu_mem_reserved_gb",
                        float(memory_summary["reserved_gb"]),
                        int(step),
                    )

            if self.tensorboard_hist_interval_steps > 0 and int(step) % self.tensorboard_hist_interval_steps == 0:
                for name, param in model.named_parameters():
                    self.tensorboard_writer.add_histogram(
                        f"params/{name}",
                        param.detach().float().cpu(),
                        int(step),
                    )
                    if param.grad is not None:
                        self.tensorboard_writer.add_histogram(
                            f"grads/{name}",
                            param.grad.detach().float().cpu(),
                            int(step),
                        )
                self.track_event("tensorboard_histograms_logged", {"step": int(step)}, global_step=step)

            self.tensorboard_writer.flush()
        except Exception as exc:
            self._safe_log_exception("[Trainer] Warning: Failed to write TensorBoard events", exc)

    def finalize(
        self,
        *,
        status: str,
        error: Optional[str],
        global_step: int,
        best_train_loss: Optional[float],
        best_val_loss: Optional[float],
        final_train_loss: Optional[float],
        final_val_loss: Optional[float],
    ) -> None:
        if self._vertex_tracking_finalized:
            return
        self._vertex_tracking_finalized = True
        if not self.enabled:
            return

        payload: Dict[str, Any] = {
            "status": status,
            "error": error,
            "global_step": int(global_step),
            "best_train_loss": best_train_loss,
            "best_val_loss": best_val_loss,
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
        }
        self.track_event("run_end", payload, global_step=global_step)

        final_metrics = {}
        if final_train_loss is not None:
            final_metrics["final_train_loss"] = float(final_train_loss)
        if final_val_loss is not None:
            final_metrics["final_val_loss"] = float(final_val_loss)
        if final_metrics:
            self.log_metrics(metrics=final_metrics, step=global_step, force=True)

        if self._vertex_run_started and self._vertex_sdk is not None:
            try:
                self._vertex_sdk.end_run()
            except Exception as exc:
                self._safe_log_exception("[Trainer] Warning: Failed to end Vertex run", exc)
        self._close_tensorboard()
