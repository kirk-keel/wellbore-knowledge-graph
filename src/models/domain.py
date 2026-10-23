"""Domain models for Wellbore Knowledge Graph entities and relationships."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class WellEntity(BaseModel):
    """Represents an oil & gas well or wellbore trajectory."""
    name: str = Field(..., description="Name or identifier of the well, e.g. 'Titan-101' or 'Well 104-B'")
    api_number: Optional[str] = Field(None, description="API well number if available, e.g. '42-123-45678'")
    field: Optional[str] = Field(None, description="Geographic field or basin, e.g. 'North Sea Alvheim' or 'Eagle Ford'")
    water_depth_ft: Optional[float] = Field(None, description="Water depth in feet for offshore wells")
    total_depth_ft: Optional[float] = Field(None, description="Total planned or drilled depth in feet MD/TVD")
    operator: Optional[str] = Field(None, description="Operating company name")


class EquipmentEntity(BaseModel):
    """Represents drilling or wellbore equipment assets."""
    serial_number: str = Field(..., description="Serial number or asset tag, e.g. 'BOP-HYD-5000-08', 'MM-775-A1'")
    equipment_type: str = Field(..., description="Type of equipment, e.g. 'Blowout Preventer', 'Mud Motor', 'Drill Pipe', 'Casing'")
    manufacturer: Optional[str] = Field(None, description="Manufacturer name, e.g. 'Cameron', 'NOV', 'Baker Hughes'")
    model: Optional[str] = Field(None, description="Model designation, e.g. 'TL Ram 18-3/4 15K'")
    working_pressure_psi: Optional[float] = Field(None, description="Rated working pressure in PSI")
    component_hierarchy: Optional[str] = Field(None, description="Subassembly or system hierarchy")


class FormationEntity(BaseModel):
    """Represents a geological formation or stratigraphic interval."""
    name: str = Field(..., description="Formation name, e.g. 'Eagle Ford Shale', 'Brent Sandstone'")
    lithology: str = Field(..., description="Lithology description, e.g. 'Shale', 'Sandstone', 'Dolomite', 'Carbonate'")
    top_depth_ft: Optional[float] = Field(None, description="Top depth MD in feet")
    bottom_depth_ft: Optional[float] = Field(None, description="Bottom depth MD in feet")
    pore_pressure_ppg: Optional[float] = Field(None, description="Pore pressure gradient in PPG (pounds per gallon)")
    fracture_gradient_ppg: Optional[float] = Field(None, description="Fracture gradient in PPG")
    hazards: List[str] = Field(default_factory=list, description="Drilling hazards, e.g. 'Lost Circulation', 'Overpressured Gas', 'Sloughing Shale'")


class FailureModeEntity(BaseModel):
    """Represents a drilling or mechanical failure mode."""
    code: str = Field(..., description="Unique failure code or tag, e.g. 'FAIL-PACKOFF', 'FAIL-RAMLEAK', 'FAIL-TWISTOFF'")
    failure_type: str = Field(..., description="Classification, e.g. 'Pack-off', 'Twist-off', 'Washout', 'Kick', 'BOP Ram Leak'")
    severity: str = Field(default="High", description="Severity level: 'Critical', 'High', 'Moderate', 'Minor'")
    description: str = Field(..., description="Technical description of the failure manifestation")
    root_cause: Optional[str] = Field(None, description="Identified root cause or mechanism")
    symptoms: List[str] = Field(default_factory=list, description="Surface or downhole symptoms, e.g. 'Spike in SPP', 'Loss of rotation'")


class SOPEntity(BaseModel):
    """Represents an operating standard, IADC guideline, or operating procedure."""
    sop_id: str = Field(..., description="SOP or standard identifier, e.g. 'IADC-WCC-04', 'API-RP-53-SEC6', 'SOP-SP-774'")
    title: str = Field(..., description="Official title of the procedure")
    category: str = Field(default="Well Control", description="Category: 'Well Control', 'Drilling', 'Tripping', 'Tripping Out', 'Casing'")
    standard_reference: Optional[str] = Field(None, description="Parent standard, e.g. 'IADC Drilling Manual', 'API Spec 16D'")
    required_barriers: List[str] = Field(default_factory=list, description="Required well control barriers, e.g. 'Hydrostatic Head', 'Blind Shear Ram'")
    steps: List[str] = Field(default_factory=list, description="Key operational sequence steps")


class IncidentEntity(BaseModel):
    """Represents a historical incident or Non-Productive Time (NPT) event."""
    incident_id: str = Field(..., description="Unique incident reference, e.g. 'INC-2024-008', 'NPT-DW-TITAN-02'")
    well_name: Optional[str] = Field(None, description="Associated well name")
    date: Optional[str] = Field(None, description="Date of incident, ISO format or YYYY-MM-DD")
    npt_hours: Optional[float] = Field(None, description="Total Non-Productive Time incurred in hours")
    depth_ft: Optional[float] = Field(None, description="Measured depth at occurrence in feet")
    description: str = Field(..., description="Incident narrative summary")
    corrective_actions: List[str] = Field(default_factory=list, description="Immediate corrective actions taken")


class Relationship(BaseModel):
    """Represents a typed relationship between entities in the graph."""
    source_type: str = Field(..., description="Source node label, e.g. 'Well', 'Incident', 'FailureMode', 'Equipment'")
    source_id: str = Field(..., description="Source identifier value matching the entity's key attribute")
    relation_type: str = Field(..., description="Relationship predicate, e.g. 'OCCURRED_IN_WELL', 'EXHIBITED_FAILURE', 'MITIGATED_BY', 'PENETRATES', 'USES_EQUIPMENT'")
    target_type: str = Field(..., description="Target node label")
    target_id: str = Field(..., description="Target identifier value matching the entity's key attribute")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Relationship edge attributes")


class DocumentChunk(BaseModel):
    """Represents a parsed textual chunk from an ingested engineering document."""
    id: str = Field(..., description="Deterministic chunk identifier, e.g. 'doc123:chunk_004'")
    doc_id: str = Field(..., description="Parent document identifier")
    chunk_index: int = Field(..., description="Sequential index of the chunk")
    page: Optional[int] = Field(None, description="Page number if extracted from a paginated PDF")
    text: str = Field(..., description="Raw text content of the chunk")
    embedding: Optional[List[float]] = Field(None, description="Dense vector embedding of chunk text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata (source, filename, timestamp)")


class WellboreExtractionResult(BaseModel):
    """Container for all entities and relationships extracted from a document or chunk."""
    wells: List[WellEntity] = Field(default_factory=list)
    equipment: List[EquipmentEntity] = Field(default_factory=list)
    formations: List[FormationEntity] = Field(default_factory=list)
    failure_modes: List[FailureModeEntity] = Field(default_factory=list)
    sops: List[SOPEntity] = Field(default_factory=list)
    incidents: List[IncidentEntity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
