"""
STEP 6: PDF Report Generator
==============================
Generates a professional, downloadable audit report from the bias audit results.

Report sections:
  1. Cover Page      - Organization name, date, risk score
  2. Executive Summary - Key findings in plain language
  3. Dataset Profile   - Shape, columns, target distribution
  4. Dataset Bias      - Pre-model bias findings per attribute
  5. Model Bias        - Post-model fairness metric tables
  6. Mitigation        - Strategies applied and results
  7. Recommendations   - Actionable next steps
"""

import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.lib.colors import HexColor


# ── Colour palette ─────────────────────────────────────────────────
C_DARK    = HexColor("#0F172A")   # near-black
C_PRIMARY = HexColor("#1D4ED8")   # strong blue
C_ACCENT  = HexColor("#7C3AED")   # violet
C_RED     = HexColor("#DC2626")
C_YELLOW  = HexColor("#D97706")
C_GREEN   = HexColor("#16A34A")
C_LIGHT   = HexColor("#F1F5F9")
C_MID     = HexColor("#CBD5E1")
C_WHITE   = HexColor("#FFFFFF")


def build_styles():
    base = getSampleStyleSheet()
    styles = {}

    styles["cover_title"] = ParagraphStyle(
        "cover_title", fontSize=28, leading=34,
        textColor=C_WHITE, fontName="Helvetica-Bold", alignment=TA_CENTER
    )
    styles["cover_sub"] = ParagraphStyle(
        "cover_sub", fontSize=13, leading=18,
        textColor=HexColor("#CBD5E1"), fontName="Helvetica", alignment=TA_CENTER
    )
    styles["cover_meta"] = ParagraphStyle(
        "cover_meta", fontSize=11, leading=16,
        textColor=HexColor("#94A3B8"), fontName="Helvetica", alignment=TA_CENTER
    )
    styles["h1"] = ParagraphStyle(
        "h1", fontSize=18, leading=22, spaceBefore=18, spaceAfter=8,
        textColor=C_PRIMARY, fontName="Helvetica-Bold"
    )
    styles["h2"] = ParagraphStyle(
        "h2", fontSize=13, leading=17, spaceBefore=12, spaceAfter=4,
        textColor=C_DARK, fontName="Helvetica-Bold"
    )
    styles["body"] = ParagraphStyle(
        "body", fontSize=10, leading=15, spaceAfter=6,
        textColor=C_DARK, fontName="Helvetica"
    )
    styles["flag_high"] = ParagraphStyle(
        "flag_high", fontSize=9.5, leading=14, spaceAfter=4,
        textColor=C_RED, fontName="Helvetica", leftIndent=12
    )
    styles["flag_med"] = ParagraphStyle(
        "flag_med", fontSize=9.5, leading=14, spaceAfter=4,
        textColor=C_YELLOW, fontName="Helvetica", leftIndent=12
    )
    styles["flag_info"] = ParagraphStyle(
        "flag_info", fontSize=9.5, leading=14, spaceAfter=4,
        textColor=C_PRIMARY, fontName="Helvetica", leftIndent=12
    )
    styles["small"] = ParagraphStyle(
        "small", fontSize=8.5, leading=12, textColor=HexColor("#64748B"),
        fontName="Helvetica"
    )
    styles["rec"] = ParagraphStyle(
        "rec", fontSize=10, leading=15, spaceAfter=5,
        textColor=C_DARK, fontName="Helvetica", leftIndent=16
    )
    return styles


# ── Helpers ──────────────────────────────────────────────────────────
def severity_color(sev):
    return {
        "SEVERE": C_RED,
        "MODERATE": C_YELLOW,
        "FAIR": C_GREEN
    }.get(sev, C_MID)


def severity_label(sev):
    return {"SEVERE": "🔴  SEVERE", "MODERATE": "🟡  MODERATE", "FAIR": "🟢  FAIR"}.get(sev, sev)


def pct(val):
    """Format a 0-1 float as a percentage string."""
    try:
        return f"{float(val)*100:.1f}%"
    except Exception:
        return str(val)


def fmt(val, decimals=3):
    try:
        return f"{float(val):.{decimals}f}"
    except Exception:
        return str(val)


def metric_table(data_rows, col_widths=None, header_row=None):
    """Build a styled reportlab Table."""
    rows = []
    if header_row:
        rows.append(header_row)
    rows.extend(data_rows)

    tbl = Table(rows, colWidths=col_widths or [2.2*inch, 1.4*inch, 1.4*inch, 1.4*inch])
    style = [
        ("BACKGROUND",  (0, 0), (-1, 0),  C_PRIMARY),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  C_WHITE),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_LIGHT]),
        ("GRID",        (0, 0), (-1, -1), 0.4, C_MID),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",(0, 0), (-1, -1), 8),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0,0), (-1, -1), 5),
        ("ALIGN",       (1, 0), (-1, -1), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
    ]
    tbl.setStyle(TableStyle(style))
    return tbl


# ══════════════════════════════════════════════════════════════════════
# MAIN REPORT CLASS
# ══════════════════════════════════════════════════════════════════════
class BiasAuditReport:

    def __init__(self, audit_results: dict, mitigation_results: dict = None,
                 org_name: str = "Organization", output_path: str = None):
        self.audit = audit_results
        self.mitigation = mitigation_results or {}
        self.org_name = org_name
        self.styles = build_styles()
        self.output_path = output_path or "/tmp/bias_audit_report.pdf"
        self.story = []
        self.W, self.H = letter

    # ── Build entire PDF ─────────────────────────────────────────────
    def generate(self) -> str:
        doc = SimpleDocTemplate(
            self.output_path,
            pagesize=letter,
            leftMargin=0.75*inch, rightMargin=0.75*inch,
            topMargin=0.6*inch,  bottomMargin=0.6*inch,
            title="AI Bias Audit Report",
            author="Bias Auditor"
        )

        self._cover_page()
        self._executive_summary()
        self._dataset_profile_section()
        self._dataset_bias_section()
        self._model_bias_section()
        if self.mitigation:
            self._mitigation_section()
        self._recommendations_section()

        doc.build(self.story, onFirstPage=self._first_page_bg,
                  onLaterPages=self._later_page_header)
        print(f"[✔] PDF report generated: {self.output_path}")
        return self.output_path

    # ── Page backgrounds / headers ───────────────────────────────────
    def _first_page_bg(self, canvas, doc):
        canvas.saveState()
        canvas.setFillColor(C_DARK)
        canvas.rect(0, 0, self.W, self.H, fill=1, stroke=0)
        # Accent bar
        canvas.setFillColor(C_PRIMARY)
        canvas.rect(0, self.H * 0.38, self.W, 4, fill=1, stroke=0)
        canvas.restoreState()

    def _later_page_header(self, canvas, doc):
        canvas.saveState()
        canvas.setFillColor(C_PRIMARY)
        canvas.rect(0, self.H - 36, self.W, 36, fill=1, stroke=0)
        canvas.setFillColor(C_WHITE)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawString(0.75*inch, self.H - 22, "AI BIAS AUDIT REPORT")
        canvas.drawRightString(self.W - 0.75*inch, self.H - 22,
                               f"{self.org_name}  |  {datetime.now().strftime('%B %d, %Y')}")
        # Footer
        canvas.setFillColor(C_MID)
        canvas.rect(0, 0, self.W, 28, fill=1, stroke=0)
        canvas.setFillColor(HexColor("#475569"))
        canvas.setFont("Helvetica", 8)
        canvas.drawCentredString(self.W / 2, 10, f"Page {doc.page}  •  Confidential")
        canvas.restoreState()

    # ── Section 1: Cover Page ────────────────────────────────────────
    def _cover_page(self):
        S = self.styles
        summary = self.audit.get("summary", {})
        risk = summary.get("risk_score", 0)
        level = summary.get("risk_level", "UNKNOWN")
        risk_color = {"HIGH": "#DC2626", "MEDIUM": "#D97706", "LOW": "#16A34A"}.get(level, "#94A3B8")

        self.story += [
            Spacer(1, 1.8*inch),
            Paragraph("AI BIAS & FAIRNESS", S["cover_sub"]),
            Spacer(1, 0.15*inch),
            Paragraph("AUDIT REPORT", S["cover_title"]),
            Spacer(1, 0.3*inch),
            HRFlowable(width="60%", thickness=2, color=HexColor("#7C3AED"),
                       hAlign="CENTER", spaceAfter=0.25*inch),
            Paragraph(self.org_name, S["cover_sub"]),
            Spacer(1, 0.15*inch),
            Paragraph(datetime.now().strftime("%B %d, %Y"), S["cover_meta"]),
            Spacer(1, 0.5*inch),
        ]

        # Risk Score badge
        risk_table = Table(
            [[Paragraph(f"<b>BIAS RISK SCORE</b>", ParagraphStyle(
                "rb", fontSize=10, textColor=HexColor("#94A3B8"),
                fontName="Helvetica-Bold", alignment=TA_CENTER
            ))],
             [Paragraph(f"<b>{risk:.0f} / 100</b>", ParagraphStyle(
                "rs", fontSize=32, textColor=HexColor(risk_color),
                fontName="Helvetica-Bold", alignment=TA_CENTER
             ))]],
            colWidths=[3*inch]
        )
        risk_table.setStyle(TableStyle([
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("BACKGROUND", (0,0), (-1,-1), HexColor("#1E293B")),
            ("ROUNDEDCORNERS", [10]),
            ("TOPPADDING", (0,0), (-1,-1), 2),
            ("BOTTOMPADDING", (0,0), (-1,-1), 25),
        ]))

        self.story += [risk_table, Spacer(1, 0.4*inch)]
        self.story += [
            Paragraph(
                f"<b>{len(summary.get('severe_attributes', []))} SEVERE</b>  |  "
                f"<b>{len(summary.get('moderate_attributes', []))} MODERATE</b>  |  "
                f"<b>{summary.get('total_flags', 0)} Total Flags</b>",
                ParagraphStyle("cs", fontSize=11, textColor=HexColor("#94A3B8"),
                               fontName="Helvetica", alignment=TA_CENTER)
            ),
            PageBreak()
        ]

    # ── Section 2: Executive Summary ─────────────────────────────────
    def _executive_summary(self):
        S = self.styles
        summary = self.audit.get("summary", {})
        self.story.append(Paragraph("Executive Summary", S["h1"]))
        self.story.append(HRFlowable(width="100%", thickness=1, color=C_MID, spaceAfter=10))

        rec = summary.get("recommendation", "")
        level = summary.get("risk_level", "UNKNOWN")
        level_color = {"HIGH": C_RED, "MEDIUM": C_YELLOW, "LOW": C_GREEN}.get(level, C_MID)

        self.story.append(Paragraph(
            f"This report presents the results of an automated bias and fairness audit "
            f"conducted on the AI system deployed by <b>{self.org_name}</b>. "
            f"The audit inspected both the training dataset and the trained model for "
            f"discriminatory patterns across protected demographic attributes.",
            S["body"]
        ))

        self.story.append(Spacer(1, 0.1*inch))

        # Key findings table
        sev_attrs  = summary.get("severe_attributes", [])
        mod_attrs  = summary.get("moderate_attributes", [])
        findings = [
            ["Overall Risk Level",   f"{level}"],
            ["Risk Score",           f"{summary.get('risk_score', 0):.1f} / 100"],
            ["Total Flags Raised",   str(summary.get("total_flags", 0))],
            ["Severely Biased Attributes", ", ".join(sev_attrs) if sev_attrs else "None"],
            ["Moderately Biased Attributes", ", ".join(mod_attrs) if mod_attrs else "None"],
        ]
        tbl = Table(findings, colWidths=[2.8*inch, 4*inch])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (0,-1), C_LIGHT),
            ("FONTNAME",   (0,0), (0,-1), "Helvetica-Bold"),
            ("FONTSIZE",   (0,0), (-1,-1), 9.5),
            ("GRID",       (0,0), (-1,-1), 0.4, C_MID),
            ("LEFTPADDING",(0,0), (-1,-1), 10),
            ("TOPPADDING", (0,0), (-1,-1), 6),
            ("BOTTOMPADDING",(0,0),(-1,-1), 6),
            ("TEXTCOLOR",  (1,0), (1,0), level_color),
            ("FONTNAME",   (1,0), (1,0), "Helvetica-Bold"),
        ]))
        self.story.append(tbl)
        self.story.append(Spacer(1, 0.15*inch))
        self.story.append(Paragraph(f"<b>Recommendation:</b> {rec}", S["body"]))
        self.story.append(Spacer(1, 0.2*inch))

    # ── Section 3: Dataset Profile ───────────────────────────────────
    def _dataset_profile_section(self):
        S = self.styles
        profile = self.audit.get("dataset_profile", {})
        shape = profile.get("shape", {})

        self.story.append(Paragraph("1. Dataset Profile", S["h1"]))
        self.story.append(HRFlowable(width="100%", thickness=1, color=C_MID, spaceAfter=10))
        self.story.append(Paragraph(
            f"The dataset contains <b>{shape.get('rows', '?'):,}</b> records and "
            f"<b>{shape.get('columns', '?')}</b> columns.",
            S["body"]
        ))

        # Target distribution
        target_info = profile.get("target_distribution", {})
        if target_info:
            self.story.append(Paragraph(
                f"<b>Target Column:</b> {target_info.get('column', '?')}", S["body"]
            ))
            dist_rows = [["Outcome", "Count", "Percentage"]]
            for label, pct_val in target_info.get("percentages", {}).items():
                count = target_info.get("counts", {}).get(label, "?")
                dist_rows.append([str(label), str(count), f"{pct_val}%"])
            self.story.append(metric_table(
                dist_rows[1:], header_row=dist_rows[0],
                col_widths=[2.5*inch, 2*inch, 2*inch]
            ))
            self.story.append(Spacer(1, 0.1*inch))

        # Sensitive attribute breakdown
        sens = profile.get("sensitive_attributes", {})
        for col, info in sens.items():
            self.story.append(Paragraph(f"Sensitive Attribute: <b>{col}</b>", S["h2"]))
            rows = [["Group", "Count", "Share"]]
            for grp, cnt in info.get("counts", {}).items():
                pct_val = info.get("percentages", {}).get(grp, "?")
                rows.append([str(grp), str(cnt), f"{pct_val}%"])
            self.story.append(metric_table(
                rows[1:], header_row=rows[0],
                col_widths=[2.5*inch, 2*inch, 2*inch]
            ))
            self.story.append(Spacer(1, 0.1*inch))

    # ── Section 4: Dataset Bias ──────────────────────────────────────
    def _dataset_bias_section(self):
        S = self.styles
        db = self.audit.get("dataset_bias", {})
        self.story.append(Paragraph("2. Dataset Bias Analysis", S["h1"]))
        self.story.append(HRFlowable(width="100%", thickness=1, color=C_MID, spaceAfter=10))
        self.story.append(Paragraph(
            "This section analyses discrimination present in the <i>raw historical data</i>, "
            "before any model is trained. Bias here will propagate into any model learned from this data.",
            S["body"]
        ))
        self.story.append(Spacer(1, 0.1*inch))

        for attr, result in db.items():
            sev = result.get("severity", "FAIR")
            sev_col = severity_color(sev)

            block = []
            block.append(Paragraph(f"Attribute: <b>{attr}</b>  —  {severity_label(sev)}", S["h2"]))

            # Metrics summary table
            metric_rows = [
                ["Metric", "Value", "Threshold", "Status"],
                ["Disparate Impact Ratio",
                 fmt(result.get("disparate_impact_ratio", 1)),
                 "≥ 0.80",
                 "FAIL" if result.get("disparate_impact_ratio", 1) < 0.8 else "PASS"],
                ["Statistical Parity Diff",
                 fmt(result.get("statistical_parity_difference", 0)),
                 "≤ 0.10",
                 "FAIL" if result.get("statistical_parity_difference", 0) > 0.1 else "PASS"],
                ["Chi-Square p-value",
                 fmt(result.get("chi2_p_value", 1), 4),
                 "< 0.05 = significant",
                 "SIG" if result.get("statistically_significant") else "NS"],
            ]
            tbl = Table(metric_rows, colWidths=[2.8*inch, 1.3*inch, 1.5*inch, 1*inch])
            style = TableStyle([
                ("BACKGROUND",  (0,0), (-1,0), C_PRIMARY),
                ("TEXTCOLOR",   (0,0), (-1,0), C_WHITE),
                ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTSIZE",    (0,0), (-1,-1), 9),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[C_WHITE, C_LIGHT]),
                ("GRID",        (0,0), (-1,-1), 0.4, C_MID),
                ("ALIGN",       (1,0), (-1,-1), "CENTER"),
                ("LEFTPADDING", (0,0), (-1,-1), 8),
                ("TOPPADDING",  (0,0), (-1,-1), 5),
                ("BOTTOMPADDING",(0,0),(-1,-1), 5),
            ])
            # Colour PASS/FAIL cells
            for ri, row in enumerate(metric_rows[1:], start=1):
                status = row[-1]
                if status == "FAIL":
                    style.add("TEXTCOLOR", (3, ri), (3, ri), C_RED)
                    style.add("FONTNAME",  (3, ri), (3, ri), "Helvetica-Bold")
                elif status == "PASS":
                    style.add("TEXTCOLOR", (3, ri), (3, ri), C_GREEN)
                    style.add("FONTNAME",  (3, ri), (3, ri), "Helvetica-Bold")
            tbl.setStyle(style)
            block.append(tbl)
            block.append(Spacer(1, 0.08*inch))

            # Outcome rates per group
            outcome_rates = result.get("outcome_rates", {})
            if outcome_rates:
                block.append(Paragraph("Positive Outcome Rate by Group:", S["h2"]))
                or_rows = [["Group", "Positive Rate", "Advantaged?"]]
                best = result.get("most_advantaged_group", "")
                for grp, rate in outcome_rates.items():
                    flag = "✔ Reference" if str(grp) == best else "▼ Disadvantaged"
                    or_rows.append([str(grp), pct(rate), flag])
                block.append(metric_table(
                    or_rows[1:], header_row=or_rows[0],
                    col_widths=[2.2*inch, 2*inch, 2.4*inch]
                ))
                block.append(Spacer(1, 0.08*inch))

            # Flags
            flags = result.get("flags", [])
            if flags:
                block.append(Paragraph("Flags & Warnings:", S["h2"]))
                for flag in flags:
                    sev_flag = flag.get("severity", "INFO")
                    flag_style = {"HIGH": S["flag_high"], "MEDIUM": S["flag_med"]}.get(
                        sev_flag, S["flag_info"]
                    )
                    prefix = {"HIGH": "⚠ HIGH: ", "MEDIUM": "● MEDIUM: ", "INFO": "ℹ INFO: "}.get(
                        sev_flag, ""
                    )
                    block.append(Paragraph(prefix + flag.get("message", ""), flag_style))

            block.append(Spacer(1, 0.2*inch))
            self.story.append(KeepTogether(block))

    # ── Section 5: Model Bias ────────────────────────────────────────
    def _model_bias_section(self):
        S = self.styles
        mb = self.audit.get("model_bias", {})
        self.story.append(PageBreak())
        self.story.append(Paragraph("3. Model Fairness Analysis", S["h1"]))
        self.story.append(HRFlowable(width="100%", thickness=1, color=C_MID, spaceAfter=10))
        self.story.append(Paragraph(
            "This section audits the <i>trained model's predictions</i> across demographic groups. "
            "A model that amplifies historical bias can cause real harm even when using "
            "seemingly neutral features.",
            S["body"]
        ))
        self.story.append(Spacer(1, 0.1*inch))

        for attr, result in mb.items():
            sev = result.get("severity", "FAIR")
            block = []
            block.append(Paragraph(f"Attribute: <b>{attr}</b>  —  {severity_label(sev)}", S["h2"]))

            # Fairness metrics table
            fm_rows = [
                ["Fairness Metric", "Value", "Threshold", "Status"],
                ["Disparate Impact Ratio",
                 fmt(result.get("disparate_impact_ratio", 1)),
                 "≥ 0.80",
                 "FAIL" if result.get("disparate_impact_ratio", 1) < 0.8 else "PASS"],
                ["Demographic Parity Diff",
                 fmt(result.get("demographic_parity_difference", 0)),
                 "≤ 0.10",
                 "FAIL" if result.get("demographic_parity_difference", 0) > 0.1 else "PASS"],
                ["Equalized Odds Diff",
                 fmt(result.get("equalized_odds_difference", 0)),
                 "≤ 0.10",
                 "FAIL" if result.get("equalized_odds_difference", 0) > 0.1 else "PASS"],
                ["Accuracy Equality Diff",
                 fmt(result.get("accuracy_equality_difference", 0)),
                 "≤ 0.05",
                 "FAIL" if result.get("accuracy_equality_difference", 0) > 0.05 else "PASS"],
            ]
            tbl = Table(fm_rows, colWidths=[2.8*inch, 1.3*inch, 1.5*inch, 1*inch])
            style = TableStyle([
                ("BACKGROUND",  (0,0), (-1,0), C_ACCENT),
                ("TEXTCOLOR",   (0,0), (-1,0), C_WHITE),
                ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTSIZE",    (0,0), (-1,-1), 9),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[C_WHITE, C_LIGHT]),
                ("GRID",        (0,0), (-1,-1), 0.4, C_MID),
                ("ALIGN",       (1,0), (-1,-1), "CENTER"),
                ("LEFTPADDING", (0,0), (-1,-1), 8),
                ("TOPPADDING",  (0,0), (-1,-1), 5),
                ("BOTTOMPADDING",(0,0),(-1,-1), 5),
            ])
            for ri, row in enumerate(fm_rows[1:], start=1):
                if row[-1] == "FAIL":
                    style.add("TEXTCOLOR", (3,ri), (3,ri), C_RED)
                    style.add("FONTNAME",  (3,ri), (3,ri), "Helvetica-Bold")
                elif row[-1] == "PASS":
                    style.add("TEXTCOLOR", (3,ri), (3,ri), C_GREEN)
                    style.add("FONTNAME",  (3,ri), (3,ri), "Helvetica-Bold")
            tbl.setStyle(style)
            block.append(tbl)
            block.append(Spacer(1, 0.08*inch))

            # Per-group model metrics
            gm = result.get("group_metrics", {})
            if gm:
                block.append(Paragraph("Per-Group Model Performance:", S["h2"]))
                gm_rows = [["Group", "Pred. Positive Rate", "True Pos. Rate", "Accuracy"]]
                for grp, m in gm.items():
                    gm_rows.append([
                        str(grp),
                        pct(m.get("positive_prediction_rate", 0)),
                        pct(m.get("true_positive_rate", 0)),
                        pct(m.get("accuracy", 0)),
                    ])
                block.append(metric_table(
                    gm_rows[1:], header_row=gm_rows[0],
                    col_widths=[2*inch, 2*inch, 1.8*inch, 1.8*inch]
                ))
                block.append(Spacer(1, 0.08*inch))

            # Flags
            for flag in result.get("flags", []):
                sv = flag.get("severity", "INFO")
                fs = {"HIGH": S["flag_high"], "MEDIUM": S["flag_med"]}.get(sv, S["flag_info"])
                pfx = {"HIGH": "⚠ HIGH: ", "MEDIUM": "● MEDIUM: "}.get(sv, "ℹ ")
                block.append(Paragraph(pfx + flag.get("message", ""), fs))

            block.append(Spacer(1, 0.2*inch))
            self.story.append(KeepTogether(block))

    # ── Section 6: Mitigation ────────────────────────────────────────
    def _mitigation_section(self):
        S = self.styles
        self.story.append(PageBreak())
        self.story.append(Paragraph("4. Bias Mitigation Results", S["h1"]))
        self.story.append(HRFlowable(width="100%", thickness=1, color=C_MID, spaceAfter=10))
        self.story.append(Paragraph(
            "Three mitigation strategies were applied. The table below shows the "
            "reduction in fairness gaps achieved by each approach.",
            S["body"]
        ))
        self.story.append(Spacer(1, 0.1*inch))

        strategy_labels = {
            "reweighting":          "Pre-Processing: Reweighting",
            "resampling":           "Pre-Processing: Resampling",
            "threshold_adjustment": "Post-Processing: Threshold Adjustment",
        }
        recommended = self.mitigation.get("recommended_strategy", "")

        for key, label in strategy_labels.items():
            res = self.mitigation.get("mitigation_results", {}).get(key)
            if not res:
                continue

            is_best = key == recommended
            block = []
            title = f"{'★ RECOMMENDED: ' if is_best else ''}{label}"
            block.append(Paragraph(title, S["h2"]))
            block.append(Paragraph(res.get("description", ""), S["body"]))
            block.append(Paragraph(
                f"<b>Overall Accuracy after mitigation:</b> "
                f"{pct(res.get('overall_accuracy', 0))}",
                S["body"]
            ))

            fi = res.get("fairness_improvement", {})
            if fi:
                fi_rows = [["Attribute", "Parity Gap (after)", "Disparate Impact", "Severity"]]
                for col, s in fi.items():
                    fi_rows.append([
                        col,
                        pct(s.get("demographic_parity_gap", 0)),
                        fmt(s.get("disparate_impact_ratio", 1)),
                        s.get("severity_after_mitigation", "?")
                    ])
                block.append(metric_table(
                    fi_rows[1:], header_row=fi_rows[0],
                    col_widths=[1.8*inch, 1.8*inch, 1.8*inch, 1.8*inch]
                ))

            block.append(Spacer(1, 0.2*inch))
            self.story.append(KeepTogether(block))

    # ── Section 7: Recommendations ───────────────────────────────────
    def _recommendations_section(self):
        S = self.styles
        self.story.append(PageBreak())
        self.story.append(Paragraph("5. Recommendations", S["h1"]))
        self.story.append(HRFlowable(width="100%", thickness=1, color=C_MID, spaceAfter=10))

        summary = self.audit.get("summary", {})
        level   = summary.get("risk_level", "LOW")
        sev_attrs = summary.get("severe_attributes", [])
        mod_attrs = summary.get("moderate_attributes", [])

        recs = []
        if level == "HIGH":
            recs.append(("IMMEDIATE", "Do not deploy or expand this system until bias is addressed. "
                          "Current bias levels may violate anti-discrimination regulations (ECOA, Title VII, GDPR)."))
        if sev_attrs:
            recs.append(("DATA COLLECTION",
                          f"Audit historical data collection processes for attributes: "
                          f"{', '.join(sev_attrs)}. Consider collecting new, more representative data."))
        recs.append(("MITIGATION",
                      "Apply the recommended mitigation strategy identified in Section 4. "
                      "Re-run this audit after applying fixes to confirm improvement."))
        recs.append(("MONITORING",
                      "Establish ongoing fairness monitoring. Bias can drift over time as "
                      "data distributions change. Re-audit at least quarterly."))
        recs.append(("TRANSPARENCY",
                      "Document all bias findings and mitigation steps. Share results with "
                      "affected communities and regulatory bodies as required."))
        recs.append(("HUMAN OVERSIGHT",
                      "Ensure high-stakes decisions (loan denials, hiring rejections) have a "
                      "clear human review process for individuals who wish to appeal."))
        recs.append(("LEGAL REVIEW",
                      "Consult with legal counsel to assess exposure under applicable "
                      "anti-discrimination laws in your jurisdiction."))

        for i, (title, text) in enumerate(recs, 1):
            self.story.append(Paragraph(
                f"<b>{i}. {title}</b>",
                ParagraphStyle("rh", fontSize=11, leading=15, spaceBefore=10,
                               textColor=C_PRIMARY, fontName="Helvetica-Bold")
            ))
            self.story.append(Paragraph(text, S["rec"]))

        self.story.append(Spacer(1, 0.3*inch))
        self.story.append(HRFlowable(width="100%", thickness=1, color=C_MID))
        self.story.append(Spacer(1, 0.1*inch))
        self.story.append(Paragraph(
            f"Report generated automatically by the AI Bias Auditor  •  "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M')}  •  "
            f"This report is confidential and intended solely for internal use.",
            S["small"]
        ))