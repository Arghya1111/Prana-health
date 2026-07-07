"""Shared modality page helpers."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from app.dashboard.components import confidence_gauge, module_status_card, patient_selector
from app.dashboard.data import load_fusion_df, load_triage_df


def get_patient_list() -> list[str]:
    fusion = load_fusion_df(5)
    triage = load_triage_df(5)
    src = fusion if not fusion.empty else triage
    if src.empty:
        return []
    return sorted(src["patient_id"].unique().tolist())


def get_patient_row(patient_id: str) -> Optional[pd.Series]:
    fusion = load_fusion_df(5)
    if not fusion.empty:
        rows = fusion[fusion["patient_id"] == patient_id]
        if not rows.empty:
            return rows.iloc[0]
    triage = load_triage_df(5)
    rows = triage[triage["patient_id"] == patient_id]
    return rows.iloc[0] if not rows.empty else None


def modality_header(module_name: str, subtitle: str) -> str:
    patients = get_patient_list()
    patient = patient_selector(patients, key=f"sel_{module_name}")
    return patient
