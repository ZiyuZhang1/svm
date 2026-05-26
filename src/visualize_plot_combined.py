import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb


def _lighten_color(color, amount=0.65):
    r, g, b = to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def _build_palette(legend_order_svm, legend_order_dnn,
                   prefix1="SVM", prefix2="DNN", lighten_amount_dnn=0.50):
    BLUE   = "#386cb0"
    PINK   = "#808080"
    ORANGE = "#52b70a"
    GREEN  = "#F57004"
    GREEN2 = "#F10A0A"

    def _strip_ppi(name):
        """Treat 'mid_fused' and 'mid_fused_ppi' as the same base."""
        n = name.lower()
        if n.endswith("_ppi"):
            n = n[:-4]
        return n

    def base_color_for_method(m, mid_toggle):
        ml = m.lower()
        # Pure PPI embeddings (ppi_emb_dw / ppi_emb_df) — keep as PINK.
        if "ppi_emb" in ml:
            return PINK
        if "early" in ml:
            return ORANGE
        if "late" in ml:
            return BLUE
        if "mid" in ml:
            return GREEN2 if (mid_toggle % 2 == 0) else GREEN
        return "#808080"

    all_base = []
    for m in legend_order_svm + legend_order_dnn:
        if m not in all_base:
            all_base.append(m)

    mid_idx = 0
    base_method_color = {}
    for m in all_base:
        base_method_color[m] = base_color_for_method(m, mid_idx)
        if "mid" in m.lower():
            mid_idx += 1

    palette = {}
    for m in legend_order_svm:
        palette[f"{prefix1} {m}"] = base_method_color[m]
    for m in legend_order_dnn:
        palette[f"{prefix2} {m}"] = _lighten_color(base_method_color[m], amount=lighten_amount_dnn)

    # Manual color overrides
    if f"{prefix2} mid_fused_ppi" in palette:
        palette[f"{prefix2} mid_fused_ppi"] = "#f88585"

    return palette


def _plot_one_setting_into_axes(
    df_long, axes, metrics_order,
    legend_order_svm, df_long2, legend_order_dnn,
    palette, prefix1="SVM", prefix2="DNN",
    group_gap_label="__GAP__",
    show_metric_titles=True,
):
    df1 = df_long.copy()
    df2 = df_long2.copy()

    df1["metric"] = pd.Categorical(df1["metric"], categories=metrics_order, ordered=True)
    df2["metric"] = pd.Categorical(df2["metric"], categories=metrics_order, ordered=True)
    metrics = list(metrics_order)

    df1["method_base"] = df1["method"].astype(str)
    df2["method_base"] = df2["method"].astype(str)
    df1["legend_label"] = prefix1 + " " + df1["method_base"]
    df2["legend_label"] = prefix2 + " " + df2["method_base"]
    df1["method_disp"] = df1["legend_label"]
    df2["method_disp"] = df2["legend_label"]

    gap_size = 3
    gap_labels = [f"{group_gap_label}_{i}" for i in range(gap_size)]
    y_order = (
        [f"{prefix1} {m}" for m in legend_order_svm] +
        gap_labels +
        [f"{prefix2} {m}" for m in legend_order_dnn]
    )
    legend_order_labels = [f"{prefix1} {m}" for m in legend_order_svm] + \
                          [f"{prefix2} {m}" for m in legend_order_dnn]

    for i, metric in enumerate(metrics):
        ax = axes[i]
        sub1 = df1[df1["metric"] == metric]
        sub2 = df2[df2["metric"] == metric]

        dummy = pd.DataFrame({
            "method_disp": [group_gap_label],
            "value": [0.0],
            "legend_label": [legend_order_labels[0]]
        })

        if not sub1.empty:
            sns.boxplot(
                data=pd.concat([sub1[["method_disp", "value", "legend_label"]], dummy], ignore_index=True),
                y="method_disp", x="value",
                order=y_order, hue="legend_label", hue_order=legend_order_labels,
                palette=palette, saturation=1.0, dodge=False, legend=False, showfliers=False, ax=ax
            )
        if not sub2.empty:
            sns.boxplot(
                data=pd.concat([sub2[["method_disp", "value", "legend_label"]], dummy], ignore_index=True),
                y="method_disp", x="value",
                order=y_order, hue="legend_label", hue_order=legend_order_labels,
                palette=palette, saturation=1.0, dodge=False, legend=False, showfliers=False, ax=ax
            )

        # --- hatch the ppi_emb_* boxes ---
        from matplotlib.patches import PathPatch
        from matplotlib.colors import to_rgba

        hatch_target_labels = [
            f"{prefix1} ppi_emb_dw",
            f"{prefix2} ppi_emb_df",
        ]
        target_colors = {
            to_rgba(palette[lbl]) for lbl in hatch_target_labels if lbl in palette
        }
        for patch in ax.patches:
            if isinstance(patch, PathPatch):
                if tuple(patch.get_facecolor()) in target_colors:
                    patch.set_hatch("///")
                    patch.set_edgecolor("black")

        if show_metric_titles:
            ax.set_title(metric, fontsize=13)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(axis="x", alpha=0.25)
        ax.tick_params(axis="x", labelsize=10)
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)


def plot_all_diseases_combined(
    disease_data,
    metrics_order,
    prefix1="SVM",
    prefix2="DNN",
):
    """
    One single figure containing every disease.

    Each disease occupies one row: n metric subplots (all_fused) + spacer + n metric subplots (ppi_fused).
    A single shared legend sits at the very top of the figure.

    disease_data: list of dicts, each with keys:
      - 'name', 'df_long_a', 'svm_a', 'df_long2_a', 'dnn_a',
        'df_long_b', 'svm_b', 'df_long2_b', 'dnn_b'
    """
    n_metrics = len(metrics_order)
    n_diseases = len(disease_data)
    if n_diseases == 0:
        return

    union_svm, union_dnn = [], []
    for d in disease_data:
        for m in d["svm_a"] + d["svm_b"]:
            if m not in union_svm:
                union_svm.append(m)
        for m in d["dnn_a"] + d["dnn_b"]:
            if m not in union_dnn:
                union_dnn.append(m)
    palette = _build_palette(union_svm, union_dnn, prefix1=prefix1, prefix2=prefix2)

    width_ratios = [1.0] * n_metrics + [0.25] + [1.0] * n_metrics
    total_cols = 2 * n_metrics + 1

    fig_w = 3.8 * (2 * n_metrics) + 0.8
    fig_h = 4.0 * n_diseases

    fig, axes = plt.subplots(
        nrows=n_diseases,
        ncols=total_cols,
        figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": width_ratios},
        sharey=False,
    )

    if n_diseases == 1:
        axes = [axes]
    else:
        axes = [list(row) for row in axes]

    for r, d in enumerate(disease_data):
        row_axes = axes[r]
        left_axes = row_axes[:n_metrics]
        spacer_ax = row_axes[n_metrics]
        right_axes = row_axes[n_metrics + 1:]
        spacer_ax.axis("off")

        _plot_one_setting_into_axes(
            d["df_long_a"], left_axes, metrics_order,
            d["svm_a"], d["df_long2_a"], d["dnn_a"],
            palette, prefix1=prefix1, prefix2=prefix2,
            show_metric_titles=True,
        )
        _plot_one_setting_into_axes(
            d["df_long_b"], right_axes, metrics_order,
            d["svm_b"], d["df_long2_b"], d["dnn_b"],
            palette, prefix1=prefix1, prefix2=prefix2,
            show_metric_titles=True,
        )

    # Shared legend at the very top
    svm_labels = [f"{prefix1} {m}" for m in union_svm]
    dnn_labels = [f"{prefix2} {m}" for m in union_dnn]
    def _make_patch(lbl):
        kwargs = dict(facecolor=palette[lbl], label=lbl)
        if lbl.endswith("ppi_emb_dw") or lbl.endswith("ppi_emb_df"):
            kwargs["hatch"] = "///"
            kwargs["edgecolor"] = "black"
        return Patch(**kwargs)

    svm_handles = [_make_patch(lbl) for lbl in svm_labels]
    dnn_handles = [_make_patch(lbl) for lbl in dnn_labels]

    ncol = max(len(svm_handles), len(dnn_handles))
    blank = Patch(facecolor=(0, 0, 0, 0), edgecolor=(0, 0, 0, 0), label="")
    svm_handles += [blank] * (ncol - len(svm_handles))
    dnn_handles += [blank] * (ncol - len(dnn_handles))

    handles = []
    for c in range(ncol):
        handles.append(svm_handles[c])
        handles.append(dnn_handles[c])

    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=ncol,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
        handlelength=1.6,
        columnspacing=1.4,
        prop={"size": 12},
    )

    plt.subplots_adjust(
        left=0.02, right=0.99,
        top=0.93, bottom=0.04,
        wspace=0.1, hspace=0.5,
    )

    # Disease title in the empty band above each row
    fig.canvas.draw()
    legend = fig.legends[0] if fig.legends else None
    if legend is not None:
        legend_bbox = legend.get_window_extent().transformed(fig.transFigure.inverted())
        legend_bottom = legend_bbox.y0
    else:
        legend_bottom = 1.0

    for r, d in enumerate(disease_data):
        row_axes = axes[r]
        row_top = max(ax.get_position().y1 for ax in row_axes)
        if r == 0:
            band_top = legend_bottom
        else:
            prev_row_axes = axes[r - 1]
            band_top = min(ax.get_position().y0 for ax in prev_row_axes)
        gap = band_top - row_top
        title_y = row_top + 0.35 * gap

        name = str(d.get("name", ""))
        display_name = name[:-3] if len(name) > 3 and name.endswith(")") else name
        fig.text(
            0.5, title_y,
            display_name,
            ha="center", va="center",
            fontsize=15, fontweight="bold",
        )
    plt.show()
    return fig