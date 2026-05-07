from __future__ import annotations

import math
import textwrap
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .faiss_index import MultimodalFaissIndex, aggregate_embedding_hits
from .render import PageRenderer
from .schemas import PageRecord, QuestionRecord
from .utils import ensure_dir


def _import_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for report visualizations.") from exc
    return plt


def _import_image():
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise ImportError("Pillow is required for gallery exports.") from exc
    return Image, ImageDraw


def fixed_crop_box(image_width: int, image_height: int, crop_name: str) -> tuple[int, int, int, int] | None:
    x_mid = image_width // 2
    y_mid = image_height // 2
    fixed_boxes = {
        "crop_r0_c0.png": (0, 0, x_mid, y_mid),
        "crop_r0_c1.png": (x_mid, 0, image_width, y_mid),
        "crop_r1_c0.png": (0, y_mid, x_mid, image_height),
        "crop_r1_c1.png": (x_mid, y_mid, image_width, image_height),
    }
    return fixed_boxes.get(crop_name)


def matched_crop_box(
    page_record: PageRecord,
    crop_name: str | None,
    renderer: PageRenderer,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int] | None:
    if not crop_name:
        return None
    if crop_name.startswith("crop_r"):
        return fixed_crop_box(image_width, image_height, crop_name)
    if crop_name.startswith("layout_"):
        for file_name, box in renderer._layout_crop_specs(page_record):  # noqa: SLF001 - visualization helper only
            if file_name == crop_name:
                return box
    return None


def render_page_thumbnail(
    page_record: PageRecord,
    renderer: PageRenderer,
    *,
    crop_name: str | None = None,
    is_gold: bool = False,
    outline_color: str = "#c63d2f",
) -> "Image.Image":
    Image, ImageDraw = _import_image()

    image_path = renderer.render_page(page_record)
    with Image.open(image_path) as source:
        image = source.convert("RGB")

    draw = ImageDraw.Draw(image)
    width, height = image.size
    highlight_box = matched_crop_box(page_record, crop_name, renderer, width, height)
    if highlight_box is not None:
        draw.rectangle(highlight_box, outline=outline_color, width=max(4, min(width, height) // 80))
    if is_gold:
        inset = max(6, min(width, height) // 100)
        draw.rectangle(
            (inset, inset, width - inset, height - inset),
            outline="#159947",
            width=max(4, min(width, height) // 90),
        )
    return image


def _wrapped(text: str, width: int = 42) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def export_query_comparison_gallery(
    *,
    question: QuestionRecord,
    manifest_lookup: Mapping[tuple[str, int], PageRecord],
    ranked_rows_by_run: Mapping[str, Sequence[dict]],
    render_cache: str | Path,
    output_path: str | Path,
    dpi: int = 150,
    top_n: int = 3,
) -> Path:
    plt = _import_matplotlib()
    renderer = PageRenderer(render_cache, dpi=dpi)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gold_keys = []
    for reference in question.gold_references:
        if reference.doc_id and reference.page_num is not None:
            key = (reference.doc_id, reference.page_num)
            if key in manifest_lookup and key not in gold_keys:
                gold_keys.append(key)

    row_specs: list[tuple[str, list[dict]]] = []
    if gold_keys:
        row_specs.append(
            (
                "Gold",
                [
                    {
                        "doc_id": doc_id,
                        "page_num": page_num,
                        "score": None,
                        "rank": index + 1,
                        "matched_crop": None,
                    }
                    for index, (doc_id, page_num) in enumerate(gold_keys[:top_n])
                ],
            )
        )

    for run_name, rows in ranked_rows_by_run.items():
        row_specs.append((run_name, list(rows[:top_n])))

    if not row_specs:
        raise ValueError("No rows were provided for the query gallery.")

    ncols = max(top_n, max(len(rows) for _, rows in row_specs))
    nrows = len(row_specs)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.2 * ncols, 3.8 * nrows),
        squeeze=False,
    )
    fig.suptitle(
        f"{question.uid}: {_wrapped(question.question, width=70)}",
        fontsize=14,
        y=0.995,
    )

    gold_key_set = set(gold_keys)
    for row_index, (label, rows) in enumerate(row_specs):
        for col_index in range(ncols):
            ax = axes[row_index][col_index]
            ax.axis("off")
            if col_index >= len(rows):
                continue

            row = rows[col_index]
            page_key = (row["doc_id"], int(row["page_num"]))
            page_record = manifest_lookup.get(page_key)
            if page_record is None:
                ax.text(0.5, 0.5, f"Missing page\n{page_key[0]}:{page_key[1]}", ha="center", va="center")
                continue

            image = render_page_thumbnail(
                page_record,
                renderer,
                crop_name=row.get("matched_crop"),
                is_gold=(page_key in gold_key_set),
            )
            ax.imshow(image)

            title_bits = [f"{row['doc_id']}:{row['page_num']}"]
            if row.get("rank") is not None:
                title_bits.insert(0, f"#{row['rank']}")
            if row.get("score") is not None:
                title_bits.append(f"s={float(row['score']):.3f}")
            if row.get("matched_crop"):
                title_bits.append(str(row["matched_crop"]).replace(".png", ""))
            ax.set_title(_wrapped(" | ".join(title_bits), width=24), fontsize=9)

            if col_index == 0:
                ax.text(
                    -0.03,
                    0.5,
                    label,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=11,
                    fontweight="bold",
                )

    plt.tight_layout(rect=(0.03, 0.03, 1, 0.95))
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def export_nearest_neighbor_gallery(
    *,
    index_path: str | Path,
    manifest_lookup: Mapping[tuple[str, int], PageRecord],
    render_cache: str | Path,
    output_path: str | Path,
    doc_id: str | None = None,
    page_num: int | None = None,
    crop_name: str | None = None,
    dpi: int = 150,
    top_n: int = 8,
    search_k: int = 64,
) -> Path:
    plt = _import_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    multimodal_index = MultimodalFaissIndex.load(index_path)
    renderer = PageRenderer(render_cache, dpi=dpi)

    anchor_index = 0
    if doc_id is not None and page_num is not None:
        for index, metadata in enumerate(multimodal_index.metadata):
            if metadata.doc_id == doc_id and metadata.page_num == page_num:
                if crop_name is None or metadata.crop_name == crop_name:
                    anchor_index = index
                    break
        else:
            raise KeyError(f"Could not find anchor page {doc_id}:{page_num} in {index_path}")

    anchor_meta = multimodal_index.metadata[anchor_index]
    anchor_key = anchor_meta.page_key()
    anchor_record = manifest_lookup.get(anchor_key)
    if anchor_record is None:
        raise KeyError(f"Anchor page missing from manifest lookup: {anchor_key}")

    query_vector = multimodal_index.index.reconstruct(anchor_index)
    scores, indices = multimodal_index.index.search(np.asarray([query_vector], dtype="float32"), search_k)
    aggregated_hits = aggregate_embedding_hits(
        scores=scores[0].tolist(),
        embedding_indices=indices[0].tolist(),
        metadata=multimodal_index.metadata,
    )
    neighbor_hits = [hit for hit in aggregated_hits if hit[0] != anchor_key][:top_n]

    ncols = 3
    nrows = max(1, math.ceil((len(neighbor_hits) + 1) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.2 * ncols, 3.9 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    anchor_image = render_page_thumbnail(anchor_record, renderer, crop_name=anchor_meta.crop_name)
    axes_flat[0].imshow(anchor_image)
    anchor_title = f"Anchor\n{anchor_meta.doc_id}:{anchor_meta.page_num}"
    if anchor_meta.crop_name:
        anchor_title += f"\n{anchor_meta.crop_name.replace('.png', '')}"
    axes_flat[0].set_title(anchor_title, fontsize=10)
    axes_flat[0].axis("off")

    for axis, (page_key, score, matched_crop_name) in zip(axes_flat[1:], neighbor_hits, strict=False):
        axis.axis("off")
        page_record = manifest_lookup.get(page_key)
        if page_record is None:
            axis.text(0.5, 0.5, f"Missing page\n{page_key[0]}:{page_key[1]}", ha="center", va="center")
            continue
        image = render_page_thumbnail(page_record, renderer, crop_name=matched_crop_name)
        axis.imshow(image)
        axis.set_title(
            _wrapped(
                f"{page_key[0]}:{page_key[1]} | sim={score:.3f}"
                + (f" | {matched_crop_name.replace('.png', '')}" if matched_crop_name else ""),
                width=26,
            ),
            fontsize=9,
        )

    for axis in axes_flat[len(neighbor_hits) + 1 :]:
        axis.axis("off")

    fig.suptitle(
        f"Nearest-neighbor gallery: {Path(index_path).name}",
        fontsize=14,
        y=0.995,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def export_doc_page_gap_plot(summary_df, output_path: str | Path) -> Path:
    plt = _import_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = summary_df.copy()
    plot_df["doc_page_gap_at_10"] = plot_df["doc_recall_at_10"] - plot_df["page_recall_at_10"]
    plot_df = plot_df.sort_values("doc_page_gap_at_10", ascending=False)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(plot_df["run_name"], plot_df["doc_page_gap_at_10"], color="#4c78a8")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_ylabel("doc Recall@10 - page Recall@10")
    ax.set_title("Document-vs-page retrieval gap")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def export_first_hit_cdf(first_hit_rank_df, output_path: str | Path, max_k: int = 50) -> Path:
    plt = _import_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ks = list(range(1, max_k + 1))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    for run_name, group in first_hit_rank_df.groupby("run_name"):
        page_ranks = group["page_first_hit_rank"]
        doc_ranks = group["doc_first_hit_rank"]
        page_curve = [float((page_ranks <= k).fillna(False).mean()) for k in ks]
        doc_curve = [float((doc_ranks <= k).fillna(False).mean()) for k in ks]
        axes[0].plot(ks, page_curve, marker="o", markersize=3, label=run_name)
        axes[1].plot(ks, doc_curve, marker="o", markersize=3, label=run_name)

    axes[0].set_title("CDF of first gold page hit")
    axes[0].set_xlabel("rank k")
    axes[0].set_ylabel("fraction of queries")
    axes[0].set_ylim(0, 1.05)
    axes[1].set_title("CDF of first gold document hit")
    axes[1].set_xlabel("rank k")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def export_gallery_batch(
    *,
    uids: Sequence[str],
    questions_by_uid: Mapping[str, QuestionRecord],
    manifest_lookup: Mapping[tuple[str, int], PageRecord],
    ranked_rows_by_uid_run: Mapping[str, Mapping[str, Sequence[dict]]],
    render_cache: str | Path,
    output_dir: str | Path,
    dpi: int = 150,
    top_n: int = 3,
) -> list[Path]:
    output_dir = ensure_dir(output_dir)
    output_paths: list[Path] = []
    for uid in uids:
        question = questions_by_uid[uid]
        output_path = output_dir / f"{uid}_query_gallery.png"
        output_paths.append(
            export_query_comparison_gallery(
                question=question,
                manifest_lookup=manifest_lookup,
                ranked_rows_by_run=ranked_rows_by_uid_run[uid],
                render_cache=render_cache,
                output_path=output_path,
                dpi=dpi,
                top_n=top_n,
            )
        )
    return output_paths
