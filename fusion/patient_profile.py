"""
Patient profile aggregation for the PRĀNA Fusion Engine.

Combines standardized predictions from every diagnostic module into a single
clinical patient object used by risk scoring, severity assessment, and reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Union

from fusion.utils import (
    ModuleName,
    ModuleResult,
    StandardizationError,
    standardize_batch,
    standardize_module_output,
    utc_now_iso,
)


@dataclass
class PatientProfile:
    """
    Unified patient object containing all available module predictions.

    Attributes
    ----------
    patient_id:
        Unique patient identifier.
    timestamp:
        ISO-8601 UTC timestamp for when the profile was assembled.
    modules:
        Mapping of module name → standardized ``ModuleResult``.
    metadata:
        Optional ancillary data (age, sex, ward, etc.).
    """

    patient_id: str
    timestamp: str = field(default_factory=utc_now_iso)
    modules: Dict[str, ModuleResult] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Construction helpers ─────────────────────────────────────────────

    @classmethod
    def from_module_outputs(
        cls,
        patient_id: str,
        outputs: Dict[str, Union[Dict[str, Any], ModuleResult, None]],
        *,
        timestamp: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PatientProfile:
        """
        Build a profile from a dict of raw or standardized module outputs.

        Parameters
        ----------
        patient_id:
            Patient identifier.
        outputs:
            Keys are module names (e.g. ``"ECG"``, ``"SpO2"``); values are
            raw prediction dicts, ``ModuleResult`` instances, or ``None``.
        """
        standardized = standardize_batch(outputs)
        return cls(
            patient_id=patient_id,
            timestamp=timestamp or utc_now_iso(),
            modules=standardized,
            metadata=metadata or {},
        )

    @classmethod
    def from_standardized_results(
        cls,
        patient_id: str,
        results: List[ModuleResult],
        *,
        timestamp: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PatientProfile:
        """Build a profile from a list of already-standardized module results."""
        modules = {result.module.upper(): result for result in results}
        return cls(
            patient_id=patient_id,
            timestamp=timestamp or utc_now_iso(),
            modules=modules,
            metadata=metadata or {},
        )

    def add_module(
        self,
        raw_or_result: Union[Dict[str, Any], ModuleResult],
        module: Optional[str] = None,
    ) -> None:
        """Add or replace a single module result on the profile."""
        if isinstance(raw_or_result, ModuleResult):
            result = raw_or_result
        else:
            result = standardize_module_output(raw_or_result, module=module)
        self.modules[result.module.upper()] = result

    # ── Accessors ──────────────────────────────────────────────────────────

    def get(self, module: str) -> Optional[ModuleResult]:
        """Return the standardized result for a module, or ``None``."""
        return self.modules.get(module.upper())

    def require(self, module: str) -> ModuleResult:
        """Return a module result or raise ``KeyError``."""
        result = self.get(module)
        if result is None:
            raise KeyError(f"Module '{module}' is not present in patient profile.")
        return result

    def available_modules(self) -> List[str]:
        """Return sorted list of module names present in the profile."""
        return sorted(self.modules.keys())

    def iter_modules(self) -> Iterator[ModuleResult]:
        """Iterate over all module results."""
        yield from self.modules.values()

    def module_count(self) -> int:
        """Return the number of modules in the profile."""
        return len(self.modules)

    def has_module(self, module: str) -> bool:
        """Return True if the given module is present."""
        return module.upper() in self.modules

    # ── Convenience property accessors ─────────────────────────────────────

    @property
    def ecg(self) -> Optional[ModuleResult]:
        return self.get(ModuleName.ECG.value)

    @property
    def eeg(self) -> Optional[ModuleResult]:
        return self.get(ModuleName.EEG.value)

    @property
    def ppg(self) -> Optional[ModuleResult]:
        return self.get(ModuleName.PPG.value)

    @property
    def spo2(self) -> Optional[ModuleResult]:
        return self.get(ModuleName.SPO2.value)

    @property
    def thermal(self) -> Optional[ModuleResult]:
        return self.get(ModuleName.THERMAL.value)

    @property
    def optical(self) -> Optional[ModuleResult]:
        return self.get(ModuleName.OPTICAL.value)

    @property
    def xray(self) -> Optional[ModuleResult]:
        return self.get(ModuleName.XRAY.value)

    # ── Serialization ──────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full patient profile to a JSON-compatible dict."""
        return {
            "patient_id": self.patient_id,
            "timestamp": self.timestamp,
            "module_count": self.module_count(),
            "available_modules": self.available_modules(),
            "modules": {
                name: result.to_dict() for name, result in sorted(self.modules.items())
            },
            "metadata": self.metadata,
        }

    def validate(self, require_at_least_one: bool = True) -> None:
        """
        Validate the profile has usable data.

        Raises
        ------
        StandardizationError
            If no modules are present and ``require_at_least_one`` is True.
        """
        if require_at_least_one and not self.modules:
            raise StandardizationError(
                "Patient profile must contain at least one module prediction."
            )

        for result in self.modules.values():
            if not result.module:
                raise StandardizationError("Module result is missing a module name.")
            if result.confidence < 0 or result.confidence > 100:
                raise StandardizationError(
                    f"Module {result.module} has invalid confidence: {result.confidence}"
                )
