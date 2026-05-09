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


class SubmitEmbedModal(ModalScreen[Optional[dict]]):
    """Pick model + output options for an ESM embedding job.

    Dismisses with a params dict ``{"model", "layers", "job_name"}``
    when the user clicks Submit, or ``None`` on Cancel. The actual
    remote ``mgr.submit()`` call runs on the parent screen so the
    modal can come down immediately — the first build of a fresh
    container can take 3-5 minutes (downloading the cu12 base image,
    layer rebuilds, ESMC-venv reinstall) and there's no point keeping
    the user staring at a frozen modal while it completes.
    """

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

        # Sort: ESM-C family first (modern recommendation), then
        # ESM-2 by parameter size ascending. Family is implicit in
        # each model's `name` (`ESM-C-300M` / `ESM2-650M`), so a
        # family-prefixed sort key keeps related entries adjacent.
        def _params_int(m: str) -> int:
            p = ESMEmbeddings.AVAILABLE_MODELS[m].get("params", "0")
            # Strip the unit and convert M/B to multipliers so
            # "650M" sorts above "150M" but below "1.5B".
            try:
                return int(float(p.rstrip("MmBb")) * (1_000 if p.upper().endswith("B") else 1))
            except (TypeError, ValueError):
                return 0

        def _family_rank(m: str) -> int:
            fam = ESMEmbeddings.AVAILABLE_MODELS[m].get("family", "")
            return 0 if fam == "esmc" else 1

        models = sorted(
            ESMEmbeddings.AVAILABLE_MODELS.keys(),
            key=lambda m: (_family_rank(m), _params_int(m)),
        )
        default_model = "esm2_t33_650M_UR50D" if "esm2_t33_650M_UR50D" in models else models[0]

        # Compose tabular labels: name padded to a uniform width,
        # then hidden-dim and VRAM estimate aligned in their own
        # columns. Looks like:
        #     ESM-C-300M    960d   1.5 GB
        #     ESM-C-600M   1152d   2.5 GB
        #     ESM2-8M       320d   0.1 GB
        # The right-aligned numbers make scanning by size trivial,
        # the left-aligned name keeps families grouped visually.
        name_w = max(
            len(ESMEmbeddings.AVAILABLE_MODELS[m].get("name", m))
            for m in models
        )
        options = []
        for m in models:
            info = ESMEmbeddings.AVAILABLE_MODELS[m]
            name = info.get("name", m).ljust(name_w)
            dim = f"{info.get('hidden_dim', '?'):>4}d"
            vram = f"{info.get('vram_gb', 0):>3.1f} GB"
            label = f"{name}  [dim]·[/dim] {dim}  [dim]·[/dim] {vram}"
            options.append((label, m))

        with Vertical(id="modal-body"):
            yield Label(
                f"[bold]Submit ESM embeddings · {self._project.name}[/bold]",
                id="modal-title",
            )
            yield Label("Model")
            yield Select(
                options,
                value=default_model,
                id="model-select",
                allow_blank=False,
            )
            # Two compact info rows below the dropdown:
            #   1. Context — set size + hardware in one line
            #      (e.g. "6 seqs · max 576 aa · 2× 1080 Ti · 11 GB · no FA")
            #   2. Verdict — color-coded VRAM estimate + attention regime
            #      (e.g. "Est. 3.1 / 11 GB · comfortable · O(L²)")
            # Background workers fill in the worker-dependent bits as
            # they complete; the verdict line re-renders whenever the
            # model selection changes.
            yield Label("[dim]Loading…[/dim]", id="context-line")
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
            self.query_one("#context-line", Label).update(f"[red]{msg}[/red]")
        except Exception:
            pass

    def _render_set_stats(self) -> None:
        # Set-stats trigger a context-line render; GPU-info trigger
        # also calls this. Both pieces fill in independently as their
        # workers finish, so this method is idempotent and uses
        # whatever's available at call time.
        self._render_context_line()

    @staticmethod
    def _compact_gpu_name(name: str) -> str:
        """Short-form GPU name for a 64-cell modal.

        `nvidia-smi` returns names like "NVIDIA GeForce GTX 1080 Ti"
        and "NVIDIA A100-SXM4-40GB"; the brand prefix and SKU suffix
        chew up modal width and don't help the user identify the
        card. Strip the noise to "1080 Ti", "A100", "RTX 3090", "H100".
        """
        if not name:
            return "?"
        # Drop the brand prefixes.
        n = name
        for prefix in ("NVIDIA GeForce ", "NVIDIA ", "GeForce ", "Tesla "):
            if n.startswith(prefix):
                n = n[len(prefix):]
        # Strip GTX prefix (1080 Ti is unambiguously a Pascal card;
        # users don't need GTX vs RTX disambiguation if they can read
        # the model number).
        if n.startswith("GTX "):
            n = n[len("GTX "):]
        # SKU suffix on datacenter cards: "A100-SXM4-40GB" → "A100".
        # Keep the model letters + first numeric chunk only.
        for sep in ("-",):
            if sep in n:
                n = n.split(sep, 1)[0]
        # Strip "80GB HBM3" tail on H100 etc.
        n = n.split(" ")
        out = []
        for tok in n:
            if tok.upper().endswith("GB") or tok.upper() in ("HBM3", "HBM2", "HBM2E", "PCIE"):
                continue
            out.append(tok)
        return " ".join(out).strip() or name

    def _render_context_line(self) -> None:
        """Single-line summary: set size + hardware.

        Renders best-effort with whatever workers have completed:
        - Both done:    "6 seqs · max 576 aa · 2× 1080 Ti · 11 GB · no FA"
        - Set only:     "6 seqs · max 576 aa · [dim]Probing GPU…[/dim]"
        - GPU only:     "[dim]Loading set…[/dim] · 2× 1080 Ti · 11 GB · no FA"
        - Neither:      "[dim]Loading…[/dim]"
        """
        try:
            label = self.query_one("#context-line", Label)
        except Exception:
            return

        # ---- Set part ----
        if self._n_seqs is None:
            set_part = "[dim]Loading set…[/dim]"
        else:
            set_part = (
                f"[bold]{self._n_seqs:,}[/bold] seqs · "
                f"max [bold]{self._max_len:,} aa[/bold]"
            )

        # ---- GPU part ----
        if not self._gpus_queried:
            gpu_part = "[dim]Probing GPU…[/dim]"
        elif not self._gpus:
            gpu_part = (
                "[red]CPU only[/red] [dim](embeddings will be slow)[/dim]"
            )
        else:
            # Group homogeneous cards: "2× 1080 Ti", or "1× A100 + 1× 3090"
            # if mixed. Drop the per-group SM tag from this row — it
            # lives on the verdict line via the attention regime.
            by_kind: dict[str, list[dict]] = {}
            for g in self._gpus:
                key = self._compact_gpu_name(g.get("name", "?"))
                by_kind.setdefault(key, []).append(g)
            kind_chunks = []
            for short, items in by_kind.items():
                count = len(items)
                kind_chunks.append(
                    f"{count}× {short}" if count > 1 else short
                )
            mem_gb = min(g.get("total_mb", 0) for g in self._gpus) / 1024.0
            fa_tag = (
                "[green]FA[/green]" if self._flash_attn_supported
                else "[yellow]no FA[/yellow]"
            )
            gpu_part = (
                f"[bold]{' + '.join(kind_chunks)}[/bold] · "
                f"{mem_gb:.0f} GB · {fa_tag}"
            )

        label.update(f"{set_part}  ·  {gpu_part}")

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
        """Back-compat shim — both workers funnel through the merged
        context-line renderer now."""
        self._render_context_line()

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
            color, verdict = "red", "likely OOM"
        elif ratio >= 0.55:
            color, verdict = "yellow", "tight"
        else:
            color, verdict = "green", "comfortable"

        # Attention regime as a short tag — the GPU/FA detail is on
        # the context line above, so we just need the complexity tag
        # here. "PyTorch math (O(L²))" when ESM-C wants FA but the
        # hardware doesn't have it, because that's the surprising
        # case worth calling out.
        wants_flash = info.get("flash_attn", False)
        gets_flash = wants_flash and self._flash_attn_supported
        if gets_flash:
            attn_label = "O(L) FlashAttention"
        elif wants_flash and not self._flash_attn_supported:
            attn_label = "O(L²) — FA unavailable on this GPU"
        else:
            attn_label = "O(L²)"

        # Single line: "[color]Est. 3.1 / 11 GB · comfortable[/color] [dim]· O(L²)[/dim]"
        msg = (
            f"[{color}]Est. {peak_gb:.1f} / {budget:.0f} GB · "
            f"{verdict}[/{color}]  [dim]· {attn_label}[/dim]"
        )

        # Optional one-line tip when the verdict is bad.
        if ratio >= 0.55:
            family = info.get("family")
            tip = ""
            if family == "esm2" and self._flash_attn_supported:
                tip = "Try ESM-C — it stays linear with FlashAttention."
            elif family == "esm2" and not self._flash_attn_supported:
                tip = "Try ESM-C-300M — smaller, and stronger per parameter."
            elif self._max_len and self._max_len > 1500:
                tip = "Filter long outliers from the homologs panel."
            if tip:
                msg += f"\n[dim]{tip}[/dim]"
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

    def _do_submit(self) -> None:
        """Validate params on the UI thread, then dismiss with the
        params dict. The parent screen runs the actual ``mgr.submit()``
        in its own worker so the modal isn't blocking the user during
        what can be a 3-5 minute first-time container build.
        """
        model = str(self.query_one("#model-select", Select).value)
        layers_str = self.query_one("#layer-input", Input).value.strip() or "-1"
        try:
            layers = [int(x) for x in layers_str.split(",") if x.strip()]
        except ValueError:
            self._set_status(
                "[red]Layer must be int(s) — e.g. -1 or 30,33[/red]"
            )
            return
        job_name = self.query_one("#job-name", Input).value.strip()
        if not job_name:
            job_name = f"{self._project.name}_embed"

        hits_fasta = self._project.active_homologs_dir() / "sequences.fasta"
        if not hits_fasta.exists():
            self._set_status(
                "[red]No hits.fasta — run a search first.[/red]"
            )
            return

        # Hand the validated params to the parent screen and bail
        # immediately. The parent runs the slow remote submit in its
        # own worker and surfaces success / error via toast.
        self.dismiss({
            "model": model,
            "layers": layers,
            "job_name": job_name,
            "hits_fasta": str(hits_fasta),
        })

    def _set_status(self, msg: str) -> None:
        self.query_one("#status-line", Label).update(msg)
