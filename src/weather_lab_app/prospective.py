"""Read-only draft of point-in-time evaluation, restricted to saved Benchmark configurations."""
from __future__ import annotations

import gradio as gr

from weather_lab_app import results


def build(suite: str) -> None:
    """Show the exact currently saved Benchmark configuration identities; never infer new recipes."""
    catalog = results.saved_configurations(suite)
    gr.Markdown(
        "## Prospective evaluation · DRAFT\n"
        "This is a design preview, **not a running collector or a validation result**. "
        "No scheduled downloads, model inference, checkpoint changes, or archive writes occur here.\n\n"
        "Compare forecasts issued prospectively with observations that arrive later. "
        "October 2026 can only be a labelled historical replay where archived issues "
        "and their availability are independently verified; do not call it live or untouched holdout."
    )
    gr.Markdown("### Existing Benchmark configurations only")
    if catalog:
        gr.Dataframe(
            value=[
                {
                    "Saved configuration": record["label"],
                    "Configuration ID": record["summary"].get("configuration_id", ""),
                    "Checkpoint": record["config"]["model"],
                    "Target": record["config"]["prediction_target"],
                    "Conditioning": record["config"]["conditioning"],
                }
                for _, record in sorted(catalog.items(), key=lambda item: item[1]["label"])
            ],
            interactive=False,
            label="Current saved Benchmark catalog (read-only; no new variants)",
        )
    else:
        gr.Markdown("No saved Benchmark configurations installed for this suite. Nothing is substituted.")

    gr.Markdown(
        "### Proposed acquisition and scoring protocol\n"
        "1. Discover each **issued** AROME cycle and record initialization time, "
        "publication/first-available time, product and complete lead coverage. "
        "Do not assume hourly new runs or silently extend a shorter horizon to 48 h.\n"
        "2. At each eligible issue, capture the exact inputs then execute **only the "
        "saved Benchmark configurations above** with their frozen checkpoint and "
        "feature provenance. Record the model version and when inference actually ran.\n"
        "3. Persist each prediction append-only, keyed by station, issue, validity, "
        "lead, product and configuration identity. Do not recompute earlier predictions.\n"
        "4. Ingest hourly observations independently, recording observed validity, "
        "first availability and any later revisions; ingest the previous day with retries.\n"
        "5. After observations arrive, score identical eligible station/valid-time/lead "
        "pairs against raw issued AROME. Show signed gain "
        "**G = MAE(AROME) − MAE(configuration)** and |G|, plus lead-time, "
        "station, issuance-cycle and coverage diagnostics.\n"
        "6. Separate October retrospective replay from truly prospective November "
        "predictions, and audit input availability and training cutoffs before "
        "making leakage or generalization claims."
    )
    gr.Markdown(
        "**Not enabled in this draft:** scheduled execution, forecast or observation "
        "fetching, scoring, immutable persistence, backfilling, model retraining, "
        "new model variants, or production deployment."
    )
