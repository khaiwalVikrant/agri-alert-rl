"""Pydantic models — type-safe contracts shared between client and server."""

from pydantic import BaseModel, Field
from typing import Literal

# Try to import openenv-core base classes; fall back to pydantic BaseModel
try:
    from openenv.core.models import Action as _BaseAction, Observation as _BaseObservation
    OPENENV_AVAILABLE = True
except ImportError:
    _BaseAction = BaseModel
    _BaseObservation = BaseModel
    OPENENV_AVAILABLE = False

LesionPattern = Literal["none", "diamond", "gray_center", "brown_border", "environmental_stress"]
CropStage = Literal["seedling", "tillering", "booting", "heading", "ripening"]
InterventionType = Literal[
    "do_nothing", "send_alert", "apply_fungicide",
    "call_agronomist", "increase_monitoring_frequency"
]


class FieldObservation(BaseModel):
    field_id: int
    lesion_coverage: float = Field(..., ge=0.0, le=1.0)
    leaf_color_index: float = Field(..., ge=0.0, le=1.0)
    lesion_pattern: LesionPattern
    crop_stage: CropStage
    days_since_last_treatment: int = Field(..., ge=0)
    field_size_ha: float = Field(..., gt=0.0)
    disease_stage: Literal["none", "early", "mid", "late"]


class RiceBlastObservation(_BaseObservation):
    lesion_coverage: float = Field(..., ge=0.0, le=1.0)
    leaf_color_index: float = Field(..., ge=0.0, le=1.0)
    lesion_pattern: LesionPattern
    temperature: float
    humidity: float = Field(..., ge=0.0, le=1.0)
    rainfall: float = Field(..., ge=0.0)
    wind_speed: float = Field(..., ge=0.0)
    crop_stage: CropStage
    days_since_last_treatment: int = Field(..., ge=0)
    field_size_ha: float = Field(..., gt=0.0)
    fields: list[FieldObservation] = Field(default_factory=list)
    timestep: int = Field(..., ge=0)
    # Required by openenv-core serialization
    reward: float | None = Field(default=None)
    done: bool = Field(default=False)


class RiceBlastAction(_BaseAction):
    intervention: InterventionType
    target_field_id: int = Field(default=0)


# Backwards-compatible aliases used by existing code
Observation = RiceBlastObservation
Action = RiceBlastAction
