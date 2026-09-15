from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
)


DISCLAIMER = (
    "Educational decision support only. This report is not a diagnosis, "
    "prescription, treatment plan, or substitute for professional medical care."
)


def _safe(value: Any) -> str:
    if value is None:
        return "-"
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _pct(value: Any) -> str:
    try:
        return f"{float(value):.1f}%"
    except Exception:
        return "-"


def _wrap_source(name: str, url: str) -> str:
    if url:
        safe_name = _safe(name)
        safe_url = _safe(url)
        return f'<b>{safe_name}</b> - <link href="{safe_url}">{safe_url}</link>'
    return _safe(name)


def _footer(canvas, doc):
    canvas.saveState()
    width, _ = A4
    canvas.setStrokeColor(colors.HexColor("#d9e2ec"))
    canvas.line(15 * mm, 12 * mm, width - 15 * mm, 12 * mm)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.setFont("Helvetica", 7)
    canvas.drawString(15 * mm, 8 * mm, "MediReco - Educational ML healthcare report")
    canvas.drawRightString(width - 15 * mm, 8 * mm, f"Page {doc.page}")
    canvas.restoreState()


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=21, leading=25, textColor=colors.HexColor("#12355b"),
            spaceAfter=5 * mm,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle", parent=base["Normal"], fontSize=9.5, leading=13,
            textColor=colors.HexColor("#64748b"), spaceAfter=6 * mm,
        ),
        "h1": ParagraphStyle(
            "H1", parent=base["Heading1"], fontName="Helvetica-Bold",
            fontSize=13, leading=16, textColor=colors.HexColor("#12355b"),
            spaceBefore=5 * mm, spaceAfter=3 * mm,
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=10.5, leading=13, textColor=colors.HexColor("#1d4ed8"),
            spaceBefore=3 * mm, spaceAfter=2 * mm,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontSize=8.7, leading=12,
            textColor=colors.HexColor("#334155"), spaceAfter=2 * mm,
        ),
        "small": ParagraphStyle(
            "Small", parent=base["BodyText"], fontSize=7.3, leading=9.5,
            textColor=colors.HexColor("#64748b"),
        ),
        "tiny": ParagraphStyle(
            "Tiny", parent=base["BodyText"], fontSize=6.5, leading=8.2,
            textColor=colors.HexColor("#64748b"),
        ),
        "note": ParagraphStyle(
            "Note", parent=base["BodyText"], fontSize=8.2, leading=11,
            textColor=colors.HexColor("#7c2d12"),
        ),
        "center": ParagraphStyle(
            "Center", parent=base["BodyText"], fontSize=8, leading=10,
            alignment=TA_CENTER, textColor=colors.HexColor("#475569"),
        ),
    }


def _table(data, widths=None, header=True):
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#dbe4ee")),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eff6ff")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#12355b")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    t.setStyle(TableStyle(commands))
    return t


def generate_checkup_pdf(payload: dict[str, Any], analysis: dict[str, Any]) -> bytes:
    """Generate a printable PDF check-up / ML decision-support report."""
    styles = _styles()
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=14 * mm,
        bottomMargin=17 * mm,
        title="MediReco Healthcare Check-up Report",
        author="MediReco",
    )

    story = []
    generated = datetime.now().strftime("%d %B %Y, %I:%M %p")

    story.append(Paragraph("MediReco", styles["title"]))
    story.append(Paragraph("Personalized Healthcare & Medicine Recommendation System", styles["subtitle"]))
    story.append(
        _table([
            [Paragraph("Report date", styles["small"]), Paragraph(_safe(generated), styles["body"]),
             Paragraph("Report type", styles["small"]), Paragraph("ML-assisted check-up summary", styles["body"])],
        ], widths=[28*mm, 55*mm, 28*mm, 59*mm], header=False)
    )

    story.append(Paragraph("Important notice", styles["h1"]))
    story.append(Paragraph(_safe(DISCLAIMER), styles["note"]))

    # Patient information
    story.append(Paragraph("1. Patient check-up inputs", styles["h1"]))
    fields = [
        ("Patient name", payload.get("patient_name") or "Not provided", "optional PDF field"),
        ("Age", payload.get("age"), "years"),
        ("Gender", payload.get("gender"), ""),
        ("Blood pressure code", payload.get("blood_pressure"), "dataset scale 0-10"),
        ("Cholesterol code", payload.get("cholesterol_level"), "dataset scale 0-10"),
        ("Fever", payload.get("fever"), ""),
        ("Cough", payload.get("cough"), ""),
        ("Fatigue", payload.get("fatigue"), ""),
        ("Difficulty breathing", payload.get("difficulty_breathing"), ""),
        ("Pregnancy flag", "Yes" if payload.get("pregnancy") else "No", "context filter"),
        ("Known allergy / medicine", payload.get("allergy") or "None provided", "user-entered"),
    ]
    rows = [[Paragraph("Item", styles["small"]), Paragraph("Value", styles["small"]), Paragraph("Notes", styles["small"])]]
    for label, value, note in fields:
        rows.append([Paragraph(_safe(label), styles["body"]), Paragraph(_safe(value), styles["body"]), Paragraph(_safe(note), styles["small"])])
    story.append(_table(rows, widths=[53*mm, 42*mm, 75*mm]))

    # Model predictions
    predictions = analysis.get("predictions", []) or []
    risk = analysis.get("risk", {}) or {}
    story.append(Paragraph("2. Model prediction summary", styles["h1"]))
    pred_rows = [[Paragraph("Rank", styles["small"]), Paragraph("Predicted class", styles["small"]), Paragraph("Model probability", styles["small"])]]
    for p in predictions:
        pred_rows.append([
            Paragraph(_safe(p.get("rank", "-")), styles["body"]),
            Paragraph(_safe(p.get("disease", "-")), styles["body"]),
            Paragraph(_pct(p.get("percentage", 0)), styles["body"]),
        ])
    if len(pred_rows) == 1:
        pred_rows.append([Paragraph("-", styles["body"])] * 3)
    story.append(_table(pred_rows, widths=[22*mm, 100*mm, 48*mm]))

    story.append(Paragraph("Risk model", styles["h2"]))
    risk_level = risk.get("risk_level", "-")
    risk_prob = risk.get("probability_percent", None)
    story.append(Paragraph(
        f"Predicted risk class: <b>{_safe(risk_level)}</b> | Model probability: <b>{_pct(risk_prob)}</b>",
        styles["body"],
    ))
    risk_probs = risk.get("probabilities", {}) or {}
    if risk_probs:
        rrows = [[Paragraph("Risk class", styles["small"]), Paragraph("Probability", styles["small"])]]
        for k, v in risk_probs.items():
            rrows.append([Paragraph(_safe(k), styles["body"]), Paragraph(_pct(float(v) * 100), styles["body"])])
        story.append(_table(rrows, widths=[80*mm, 50*mm]))

    # Recommendations
    recommendations = analysis.get("recommendations", {}) or {}
    groups = recommendations.get("groups", []) or []
    story.append(Paragraph("3. Treatment & care reference", styles["h1"]))
    story.append(Paragraph(
        _safe(recommendations.get("message", "Evidence-linked project references.")),
        styles["body"],
    ))

    if not groups:
        story.append(Paragraph("No evidence-linked reference items were returned for these model classes.", styles["body"]))
    else:
        for group in groups:
            disease = group.get("disease", "Unknown")
            probability = group.get("probability", 0)
            story.append(Paragraph(
                f"{_safe(disease)} - {_pct(probability)} model probability",
                styles["h2"],
            ))

            meds = group.get("medicines", []) or []
            if meds:
                story.append(Paragraph("Medicine / treatment references", styles["body"]))
                mrows = [[
                    Paragraph("Reference", styles["small"]),
                    Paragraph("Details", styles["small"]),
                    Paragraph("Safety / review", styles["small"]),
                ]]
                for m in meds:
                    mrows.append([
                        Paragraph(_safe(m.get("item")), styles["body"]),
                        Paragraph(_safe(m.get("details")), styles["small"]),
                        Paragraph(_safe(m.get("safety_note")), styles["small"]),
                    ])
                story.append(_table(mrows, widths=[43*mm, 82*mm, 55*mm]))
            else:
                story.append(Paragraph("No medicine-specific reference was returned for this class.", styles["body"]))

            for section_key, heading in [
                ("support", "Other guidance"),
                ("food", "Food / lifestyle references"),
                ("when_to_seek_care", "When to seek professional care"),
                ("emergency", "Urgent / emergency information"),
                ("safety", "Safety notes"),
            ]:
                items = group.get(section_key, []) or []
                if not items:
                    continue
                story.append(Paragraph(heading, styles["body"]))
                for item in items:
                    story.append(Paragraph(
                        f"<b>{_safe(item.get('item'))}</b>: {_safe(item.get('details'))}",
                        styles["small"],
                    ))

            story.append(Spacer(1, 2 * mm))

    # Feature importance
    model_info = analysis.get("model_info", {}) or {}
    importance = model_info.get("feature_importance", []) or []
    if importance:
        story.append(Paragraph("4. Model explainability", styles["h1"]))
        story.append(Paragraph(
            "Approximate Random Forest feature importance aggregated back to the original input fields. "
            "This shows which inputs influenced the trained model most in aggregate; it is not causal evidence.",
            styles["body"],
        ))
        irows = [[Paragraph("Feature", styles["small"]), Paragraph("Relative importance", styles["small"])]]
        for row in importance[:8]:
            irows.append([Paragraph(_safe(row.get("feature")), styles["body"]), Paragraph(_pct(row.get("value")), styles["body"])])
        story.append(_table(irows, widths=[100*mm, 70*mm]))

    # Sources
    source_rows = []
    seen = set()
    for group in groups:
        for section in ["medicines", "support", "food", "when_to_seek_care", "emergency", "safety"]:
            for item in group.get(section, []) or []:
                url = str(item.get("source_url", ""))
                name = str(item.get("source_name", ""))
                key = (name, url)
                if key in seen or not (name or url):
                    continue
                seen.add(key)
                source_rows.append((name, url))

    story.append(Paragraph("5. Evidence sources used by the recommendation layer", styles["h1"]))
    if source_rows:
        for name, url in source_rows:
            story.append(Paragraph(_wrap_source(name, url), styles["small"]))
            story.append(Spacer(1, 1.5 * mm))
    else:
        story.append(Paragraph("No external source links were attached to the returned items.", styles["small"]))

    # Model details / limitations
    story.append(Paragraph("6. Model and project notes", styles["h1"]))
    metrics = model_info.get("metrics", {}) or {}
    model_lines = [
        f"Algorithm: {model_info.get('algorithm', 'Random Forest Classifier')}",
        f"Number of trees: {model_info.get('n_estimators', '-')}",
        f"Dataset rows used for training/evaluation: {metrics.get('dataset_rows', '-')}",
        f"Test accuracy: {_pct(float(metrics['disease_accuracy']) * 100) if metrics.get('disease_accuracy') is not None else '-'}",
        f"Balanced accuracy: {_pct(float(metrics['disease_balanced_accuracy']) * 100) if metrics.get('disease_balanced_accuracy') is not None else '-'}",
    ]
    for line in model_lines:
        story.append(Paragraph(_safe(line), styles["body"]))

    story.append(Paragraph(
        "Interpretation note: the underlying classroom dataset is small and imbalanced. "
        "Accuracy alone may overstate performance, so this report also shows balanced accuracy. "
        "The predictions should not be used as a clinical diagnosis.",
        styles["note"],
    ))

    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(DISCLAIMER, styles["center"]))

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buffer.getvalue()
