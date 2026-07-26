"""
Report generation utilities: produces downloadable
CSV / Excel / PDF business reports, including PDF reports
with embedded charts.
"""

import io
import logging
from datetime import datetime
from typing import List, Tuple, Union

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def to_excel_bytes(sheets: dict) -> bytes:
    """sheets: dict of {sheet_name: dataframe}"""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    return buffer.getvalue()


def make_bar_chart_image(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> bytes:
    """Render a simple matplotlib bar chart to PNG bytes for embedding in a PDF."""
    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=150)
    plot_df = df.head(12)
    ax.bar(plot_df[x_col].astype(str), plot_df[y_col], color="#4C72B0")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    plt.xticks(rotation=40, ha="right", fontsize=7)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def to_pdf_bytes(title: str, sections: list, chart: Tuple[bytes, str] = None) -> bytes:
    """
    sections: list of (heading, dataframe_or_text)
    chart: optional (png_bytes, caption) tuple to embed a chart image
    Uses reportlab to build a simple business report PDF.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Table, TableStyle, Image)
    from reportlab.lib.styles import getSampleStyleSheet

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                             topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]),
             Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]),
             Spacer(1, 0.6*cm)]

    if chart is not None:
        png_bytes, caption = chart
        img_buf = io.BytesIO(png_bytes)
        story.append(Image(img_buf, width=15*cm, height=8*cm))
        story.append(Paragraph(caption, styles["Italic"]))
        story.append(Spacer(1, 0.6*cm))

    for heading, content in sections:
        story.append(Paragraph(heading, styles["Heading2"]))
        story.append(Spacer(1, 0.2*cm))
        if isinstance(content, pd.DataFrame):
            data = [content.columns.tolist()] + content.round(2).astype(str).values.tolist()
            table = Table(data, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4C72B0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F2F2")]),
            ]))
            story.append(table)
        else:
            story.append(Paragraph(str(content), styles["Normal"]))
        story.append(Spacer(1, 0.6*cm))

    doc.build(story)
    return buffer.getvalue()
