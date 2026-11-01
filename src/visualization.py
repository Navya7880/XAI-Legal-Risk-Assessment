"""Reusable Plotly figures for application pages."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


BLUE = "#146C65"


def class_distribution(data: pd.DataFrame, target: str) -> go.Figure:
    counts = data[target].astype(str).value_counts().rename_axis(target).reset_index(name="count")
    return px.bar(counts, x=target, y="count", color=target, title="Target class distribution")


def feature_distribution(data: pd.DataFrame, feature: str) -> go.Figure:
    if pd.api.types.is_numeric_dtype(data[feature]):
        return px.histogram(data, x=feature, color_discrete_sequence=[BLUE], title=f"Distribution: {feature}")
    counts = data[feature].astype(str).value_counts().rename_axis(feature).reset_index(name="count")
    return px.bar(counts, x=feature, y="count", color=feature, title=f"Distribution: {feature}")


def confusion_figure(matrix: list[list[int]], labels: list[str]) -> go.Figure:
    return px.imshow(
        matrix,
        x=labels,
        y=labels,
        text_auto=True,
        labels={"x": "Predicted", "y": "Actual", "color": "Count"},
        title="Confusion matrix",
        color_continuous_scale="Blues",
    )
