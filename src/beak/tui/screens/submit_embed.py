"""Modal for configuring + submitting an ESM embedding job."""

from typing import Optional

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select

from ...project import BeakProject


# Hardware budget assumed for the warning thresholds. The user's lab
# server has 2× GTX 1080 Ti (11 GB each); beak embeds one sequence at
# a time on a single GPU, so per-job budget is one card. We poll
# `nvidia-smi` on modal mount to refine this when possible, but the
# fallback assumption is "modest 2018-era card" rather than an A100.
_DEFAULT_GPU_BUDGET_GB = 11.0


class SubmitEmbedModal(ModalScreen[Optional[str]]):
    """Pick model + output options for an ESM embedding job."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, project: BeakProject) -> None:
        super().__init__()
        self._project = project
        self._submitting = False
        # Filled in by `_load_set_stats` worker; consumed by
        # `_refresh_estimate` whenever the model selection changes.
        self._max_len: Optional[int] = None
        self._n_seqs: Optional[int] = None
        self._p90_len: Optional[int] = None
        self._n_long: Optional[int] = None  # how many > 1500 aa
        self._gpu_budget_gb: float = _DEFAULT_GPU_BUDGET_GB
        self._gpu_name: Optional[str] = None
        # Pre-Ampere GPUs (Pascal: 1080 Ti, P100; Volta: V100; older)
        # can't run FlashAttention even when the model code requests
        # it — SDPA silently falls back to the math backend, which is
        # quadratic in sequence length. We default to True (assume
        # modern hardware) and flip to False once `query_gpus` confirms
        # the actual compute capability is < SM 8.0; that flip drives
        # the VRAM estimator into its O(L²) branch even for ESM-C.
        self._flash_attn_supported: bool = True
        # Full per-GPU info from `query_gpus()` — kept so the modal
        # can render a one-line hardware summary that's accurate
        # whether the remote has 1× A100, 2× 1080 Ti, an H100, or
        # nothing at all (CPU-only).
        self._gpus: list[dict] = []
        # Distinguish "haven't queried yet" from "queried, found
        # nothing" so the UI can show a different message for each.
        self._gpus_queried: bool = False

    def compose(self) -> ComposeResult:
        from ...remote.embeddings import ESMEmbeddings

        models = list(ESMEmbeddings.AVAILABLE_MODELS.keys())
        default_model = "esm2_t33_650M_UR50D" if "esm2_t33_650M_UR50D" in models else models[0]

        with Vertical(id="modal-body"):
            yield Label(
                f"[bold]Submit ESM embeddings · {self._project.name}[/bold]",
                id="modal-title",
            )
            yield Label("[dim]Runs on the configured remote GPU[/dim]")
            yield Label("")

            yield Label("Model")
            yield Select(
                [(m, m) for m in models],
                value=default_model,
                id="model-select",
                allow_blank=False,
            )
            # Inline panel: "7,241 sequences · longest 1,532 aa · …"
            # Filled in by background workers after mount; updated when
            # the user changes models.
            yield Label("[dim]Loading set stats…[/dim]", id="set-stats")
            # Hardware summary row — surfaces the actual remote GPUs
            # and their FlashAttention capability so users on diverse
            # hardware (Pascal lab servers vs Hopper rented boxes)
            # see warnings calibrated to *their* setup.
            yield Label("[dim]Probing remote GPU…[/dim]", id="gpu-info")
            yield Label("", id="vram-estimate")
            yield Label("Layer (e.g. -1 for last; comma-sep for multiple)")
            yield Input(value="-1", id="layer-input")
            yield Label("Job name (optional)")
            yield Input(
                placeholder=f"{self._project.name}_embed",
                id="job-name",
            )
            yield Label("", id="status-line")

            with Horizontal(id="modal-buttons"):
                yield Button("Cancel", id="cancel-btn")
                yield Button("Submit", id="submit-btn", variant="primary")

    def on_mount(self) -> None:
        # Kick both background workers off the moment the modal opens.
        # Stats parse the FASTA (cheap, local); GPU query SSH's the
        # remote and is best-effort — if it fails, we just keep the
        # 11 GB default budget and never block the UI on it.
        self._load_set_stats()
        self._query_gpu_budget()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "model-select":
            self._refresh_estimate()

    @work(thread=True, exclusive=True, group="embed-stats")
    def _load_set_stats(self) -> None:
        """Stream the active set's hits FASTA → (n_seqs, max, p90, n_long).

        Same line-by-line approach the layer-detail length histogram
        uses — a 40 MB hits file doesn't pull into memory at once.
        """
        fasta = self._project.active_homologs_dir() / "sequences.fasta"
        lengths: list[int] = []
        try:
            cur = 0
            with open(fasta) as f:
                for line in f:
                    if line.startswith(">"):
                        if cur:
                            lengths.append(cur)
                        cur = 0
                    else:
                        cur += len(line.strip())
                if cur:
                    lengths.append(cur)
        except Exception:
            lengths = []

        if not lengths:
            self.app.call_from_thread(
                self._set_stats_unavailable,
                "No hits.fasta — run a search first.",
            )
            return

        lengths.sort()
        n = len(lengths)
        self._n_seqs = n
        self._max_len = lengths[-1]
        self._p90_len = lengths[int(0.9 * (n - 1))]
        self._n_long = sum(1 for L in lengths if L > 1500)
        self.app.call_from_thread(self._render_set_stats)
        self.app.call_from_thread(self._refresh_estimate)

    def _set_stats_unavailable(self, msg: str) -> None:
        try:
            self.query_one("#set-stats", Label).update(f"[red]{msg}[/red]")
        except Exception:
            pass

    def _render_set_stats(self) -> None:
        if self._n_seqs is None:
            return
        parts = [
            f"[bold]{self._n_seqs:,}[/bold] sequences",
            f"longest [bold]{self._max_len:,} aa[/bold]",
            f"p90 {self._p90_len:,} aa",
        ]
        if self._n_long:
            parts.append(
                f"[yellow]{self._n_long:,} over 1,500 aa[/yellow]"
            )
        try:
            self.query_one("#set-stats", Label).update(" · ".join(parts))
        except Exception:
            pass

    @work(thread=True, exclusive=True, group="embed-gpu-query")
    def _query_gpu_budget(self) -> None:
        """Poll `nvidia-smi` on the remote so the threshold reflects
        the actual hardware (e.g., 11 GB on a 1080 Ti vs 24 GB on a
        3090 vs 80 GB on an A100). Best-effort: if SSH is slow or
        unavailable, the default 11 GB stands."""
        from ...remote.embeddings import ESMEmbeddings
        mgr = None
        try:
            mgr = ESMEmbeddings()
            gpus = mgr.query_gpus()
        except Exception:
            return
        finally:
            try:
                if mgr is not None and getattr(mgr, "conn", None) is not None:
                    mgr.conn.close()
            except Exception:
                pass
        self._gpus_queried = True
        self._gpus = list(gpus)
        if not gpus:
            # CPU-only host or non-NVIDIA hardware. Surface this
            # explicitly — embeddings will run on CPU and be very
            # slow (10-100× slower than a modest GPU), so the user
            # should know before they hit submit.
            self.app.call_from_thread(self._render_gpu_info)
            self.app.call_from_thread(self._refresh_estimate)
            return
        # Use the *smallest* card's total memory: a job randomly lands
        # on either GPU, so the conservative bound is the worse one.
        total_mb = min(g["total_mb"] for g in gpus)
        self._gpu_budget_gb = total_mb / 1024.0
        smallest = min(gpus, key=lambda g: g["total_mb"])
        self._gpu_name = smallest.get("name")
        # FA support is "every card supports it" — a job may land on
        # any GPU in the pool, so the conservative bound is the
        # worst-case card. Use the authoritative compute_cap when
        # available; fall back to the name heuristic.
        self._flash_attn_supported = all(
            ESMEmbeddings.gpu_supports_flash_attn(
                gpu_name=g.get("name", ""),
                compute_cap=g.get("compute_cap", ""),
            )
            for g in gpus
        )
        self.app.call_from_thread(self._render_gpu_info)
        self.app.call_from_thread(self._refresh_estimate)

    def _render_gpu_info(self) -> None:
        """One-line summary of the remote GPU pool.

        Examples:
            2× GeForce GTX 1080 Ti · 11 GB · SM 6.1 · no FlashAttention
            1× NVIDIA H100 80GB HBM3 · 80 GB · SM 9.0 · FlashAttention
            CPU only — no NVIDIA GPU detected (embeddings will be slow)
        """
        try:
            label = self.query_one("#gpu-info", Label)
        except Exception:
            return
        if not self._gpus:
            label.update(
                "[red]CPU only[/red] — no NVIDIA GPU detected on remote. "
                "[dim]Embeddings will run on CPU, which is 10-100× "
                "slower than a modest GPU.[/dim]"
            )
            return

        # Group by (name, compute_cap) so 2× 1080 Ti collapses to a
        # single row but a mixed-card box (1× A100 + 1× 3090) shows
        # both. Common case (homogeneous pool) reads cleanly; the
        # less common case still tells the truth.
        by_kind: dict[tuple[str, str], list[dict]] = {}
        for g in self._gpus:
            key = (g.get("name", "?"), g.get("compute_cap", ""))
            by_kind.setdefault(key, []).append(g)

        parts = []
        for (name, cc), items in by_kind.items():
            count = len(items)
            mem_gb = items[0].get("total_mb", 0) / 1024.0
            cc_tag = f" · SM {cc}" if cc else ""
            count_tag = f"{count}× " if count > 1 else ""
            parts.append(
                f"[bold]{count_tag}{name}[/bold] · "
                f"{mem_gb:.0f} GB{cc_tag}"
            )
        head = "  +  ".join(parts)
        fa_tag = (
            "[green]FlashAttention[/green]"
            if self._flash_attn_supported
            else "[yellow]no FlashAttention[/yellow]"
        )
        label.update(f"{head} · {fa_tag}")

    def _refresh_estimate(self) -> None:
        """Recompute and re-render the VRAM estimate label.

        Called whenever the model selection changes or the FASTA / GPU
        workers finish. Cheap (no I/O) — just runs the formula.
        """
        if self._max_len is None:
            return
        try:
            select = self.query_one("#model-select", Select)
            label = self.query_one("#vram-estimate", Label)
        except Exception:
            return

        from ...remote.embeddings import ESMEmbeddings
        model_id = str(select.value) if select.value is not None else None
        if not model_id or model_id not in ESMEmbeddings.AVAILABLE_MODELS:
            label.update("")
            return

        info = ESMEmbeddings.AVAILABLE_MODELS[model_id]
        peak_gb = ESMEmbeddings.estimate_peak_vram_gb(
            model_id, self._max_len,
            flash_attn_available=self._flash_attn_supported,
        )
        budget = self._gpu_budget_gb
        ratio = peak_gb / budget if budget > 0 else 0.0

        # Color thresholds: leave 20% of budget for CUDA driver state,
        # the docker container's working set, and the allocator's
        # fragmentation. So 80% peak ≈ "expect OOM".
        if ratio >= 0.85:
            color, verdict = "red", "[bold]likely OOM[/bold]"
        elif ratio >= 0.55:
            color, verdict = "yellow", "tight — risk on the longest sequences"
        else:
            color, verdict = "green", "comfortable"

        gpu_tag = f" on {self._gpu_name}" if self._gpu_name else ""
        # Distinguish "model has FA code path" from "hardware can run
        # FA" — Pascal cards request FA via SDPA but get math-backend
        # at runtime, so calling it FlashAttention in the modal would
        # be misleading.
        wants_flash = info.get("flash_attn", False)
        gets_flash = wants_flash and self._flash_attn_supported
        if gets_flash:
            attn_label = "FlashAttention (O(L))"
        elif wants_flash and not self._flash_attn_supported:
            attn_label = (
                "PyTorch math attention (O(L²)) — "
                "FA disabled; GPU is pre-Ampere"
            )
        else:
            attn_label = "PyTorch attention (O(L²))"
        msg = (
            f"[{color}]Est. peak VRAM @ longest seq:[/{color}] "
            f"[bold]{peak_gb:.1f} GB[/bold] / {budget:.0f} GB{gpu_tag} "
            f"— {verdict}\n"
            f"[dim]{attn_label} · model {info['name']}[/dim]"
        )

        if ratio >= 0.55:
            # When neither family can use FA on this hardware, ESM-C
            # is no longer the magic escape from quadratic scaling —
            # it's just a smaller model. Tailor the tip accordingly.
            family = info.get("family")
            if family == "esm2" and self._flash_attn_supported:
                msg += (
                    "\n[dim]Tip: ESM-C uses FlashAttention and stays "
                    "linear in length — it'll fit where ESM-2 won't.[/dim]"
                )
            elif family == "esm2" and not self._flash_attn_supported:
                msg += (
                    "\n[dim]Tip: ESM-C-300M is smaller (and a stronger "
                    "model per parameter) — consider it. Both families "
                    "are quadratic on pre-Ampere GPUs, so longest-seq "
                    "headroom comes mostly from picking a smaller "
                    "model or filtering long outliers.[/dim]"
                )
            elif self._max_len and self._max_len > 1500:
                msg += (
                    "\n[dim]Filter long outliers from the homologs "
                    "panel — most of the cost is from the tail of "
                    "the length distribution.[/dim]"
                )
        label.update(msg)

    def action_cancel(self) -> None:
        if self._submitting:
            return
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-btn":
            self.action_cancel()
        elif event.button.id == "submit-btn" and not self._submitting:
            self._do_submit()

    @work(thread=True, exclusive=True, group="embed-submit")
    def _do_submit(self) -> None:
        model = str(self.query_one("#model-select", Select).value)
        layers_str = self.query_one("#layer-input", Input).value.strip() or "-1"
        try:
            layers = [int(x) for x in layers_str.split(",") if x.strip()]
        except ValueError:
            self.app.call_from_thread(
                self._set_status, "[red]Layer must be int(s) — e.g. -1 or 30,33[/red]"
            )
            return
        job_name = self.query_one("#job-name", Input).value.strip()
        if not job_name:
            job_name = f"{self._project.name}_embed"

        hits_fasta = self._project.active_homologs_dir() / "sequences.fasta"
        if not hits_fasta.exists():
            self.app.call_from_thread(
                self._set_status,
                "[red]No hits.fasta — run a search first.[/red]",
            )
            return

        self._submitting = True
        self.app.call_from_thread(
            self._set_status, "[dim]Submitting (a few seconds)…[/dim]"
        )

        mgr = None
        try:
            from ...remote.embeddings import ESMEmbeddings
            mgr = ESMEmbeddings()
            job_id = mgr.submit(
                str(hits_fasta),
                model=model,
                job_name=job_name,
                repr_layers=layers,
            )
            # Stamp the job onto the active homolog set's embedding
            # entry — multi-set support means every embedding is keyed
            # to the set it was computed against.
            self._project.update_active_embeddings_set(
                model=model,
                remote={"job_id": job_id},
            )
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(
                self._set_status, f"[red]{type(e).__name__}: {e}[/red]"
            )
            self._submitting = False
            return
        finally:
            try:
                if mgr is not None and getattr(mgr, "conn", None) is not None:
                    mgr.conn.close()
            except Exception:
                pass

        self.app.call_from_thread(self.dismiss, job_id)

    def _set_status(self, msg: str) -> None:
        self.query_one("#status-line", Label).update(msg)
