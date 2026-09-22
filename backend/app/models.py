from pydantic import BaseModel, Field


class Ingredient(BaseModel):
    ingredient_concept_id: int
    ingredient_name: str


class Condition(BaseModel):
    condition_concept_id: int
    condition_name: str


class PredictRequest(BaseModel):
    ingredient_concept_id: int
    condition_concept_id: int


class EvidenceRow(BaseModel):
    source: str = Field(description="faers | semmeddb | eu_label")
    relationship: str | None = None
    value: float | None = None


class PredictResponse(BaseModel):
    ingredient_concept_id: int
    ingredient_name: str
    condition_concept_id: int
    condition_name: str
    treats_score: float | None = Field(default=None, ge=0.0, le=1.0)
    causes_score: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: list[EvidenceRow]
    status: str = Field(description="ready | model_not_trained | error")
    model: str | None = None


class HealthResponse(BaseModel):
    status: str
    data_files_present: dict[str, bool]


class AnnotationCreate(BaseModel):
    ingredient_concept_id: int
    condition_concept_id: int
    assertion: str = Field(description="treats | causes")
    notes: str | None = None
    added_by: str | None = None


class Annotation(BaseModel):
    ingredient_concept_id: int
    ingredient_name: str
    condition_concept_id: int
    condition_name: str
    assertion: str
    notes: str | None = None
    added_by: str | None = None
    added_at: str


class DrugSummaryTopRow(BaseModel):
    condition_concept_id: int
    condition_name: str
    faers_prr: float | None = None
    semmeddb_relationships: str | None = None


class DrugSummaryResponse(BaseModel):
    ingredient_concept_id: int
    rows_scanned: int
    conditions: int
    faers_pairs: int
    semmeddb_pairs: int
    top: list[DrugSummaryTopRow]
    elapsed_ms: int
    executor: str
