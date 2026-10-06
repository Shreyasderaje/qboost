"""Task registry for the three diagnostic screening workloads.

Each task is a binary screening problem.  The feature specs drive three
things: the synthetic data generator, the clinic console UI (input ranges)
and the explanation panel (feature direction).  When real datasets are
loaded through ``backend.data.loaders`` they are mapped onto the same
feature schema so the rest of the stack is agnostic to the data source.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    label: str
    unit: str
    lo: float
    hi: float
    direction: int  # +1: higher value raises risk, -1: lower value raises risk


@dataclass(frozen=True)
class TaskSpec:
    key: str
    display_name: str
    subtitle: str
    positive_label: str
    negative_label: str
    prevalence: float
    features: list[FeatureSpec] = field(default_factory=list)

    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self.features]

    @property
    def n_features(self) -> int:
        return len(self.features)

    def spec(self, name: str) -> FeatureSpec:
        for f in self.features:
            if f.name == name:
                return f
        raise KeyError(name)


TASKS: dict[str, TaskSpec] = {
    "tb": TaskSpec(
        key="tb",
        display_name="Tuberculosis",
        subtitle="Chest X-ray screening",
        positive_label="TB findings present",
        negative_label="No TB findings",
        prevalence=0.38,
        features=[
            FeatureSpec("cavity_score", "Cavity score", "index", 0.0, 100.0, +1),
            FeatureSpec("infiltrate_density", "Infiltrate density", "index", 0.0, 100.0, +1),
            FeatureSpec("opacity_volume", "Opacity volume", "index", 0.0, 100.0, +1),
            FeatureSpec("pleural_effusion", "Pleural effusion", "index", 0.0, 100.0, +1),
            FeatureSpec("hilar_fullness", "Hilar fullness", "index", 0.0, 100.0, +1),
            FeatureSpec("nodule_count", "Nodule count", "count", 0.0, 20.0, +1),
            FeatureSpec("diaphragm_irregularity", "Diaphragm irregularity", "index", 0.0, 100.0, +1),
            FeatureSpec("lung_field_asymmetry", "Lung field asymmetry", "index", 0.0, 100.0, +1),
            FeatureSpec("bronchovascular_marking", "Bronchovascular marking", "index", 0.0, 100.0, +1),
            FeatureSpec("costophrenic_blunting", "Costophrenic blunting", "index", 0.0, 100.0, +1),
        ],
    ),
    "malaria": TaskSpec(
        key="malaria",
        display_name="Malaria",
        subtitle="Blood smear analysis",
        positive_label="Parasites detected",
        negative_label="Uninfected smear",
        prevalence=0.42,
        features=[
            FeatureSpec("parasite_density", "Parasite density", "index", 0.0, 100.0, +1),
            FeatureSpec("ring_form_score", "Ring-form score", "index", 0.0, 100.0, +1),
            FeatureSpec("chromatin_pattern", "Chromatin pattern anomaly", "index", 0.0, 100.0, +1),
            FeatureSpec("cytoplasm_stain", "Cytoplasm stain deviation", "index", 0.0, 100.0, +1),
            FeatureSpec("cell_irregularity", "Cell irregularity", "index", 0.0, 100.0, +1),
            FeatureSpec("gametocyte_score", "Gametocyte score", "index", 0.0, 100.0, +1),
            FeatureSpec("cell_size_deviation", "Cell size deviation", "index", 0.0, 100.0, +1),
            FeatureSpec("background_roughness", "Background roughness", "index", 0.0, 100.0, +1),
        ],
    ),
    "diabetes": TaskSpec(
        key="diabetes",
        display_name="Diabetes risk",
        subtitle="Metabolic risk scoring",
        positive_label="High risk",
        negative_label="Low risk",
        prevalence=0.35,
        features=[
            FeatureSpec("glucose", "Plasma glucose", "mg/dL", 60.0, 300.0, +1),
            FeatureSpec("bmi", "Body mass index", "kg/m²", 15.0, 60.0, +1),
            FeatureSpec("age", "Age", "years", 21.0, 81.0, +1),
            FeatureSpec("insulin", "Serum insulin", "µU/mL", 14.0, 400.0, +1),
            FeatureSpec("blood_pressure", "Diastolic blood pressure", "mmHg", 40.0, 120.0, +1),
            FeatureSpec("pregnancies", "Pregnancies", "count", 0.0, 15.0, +1),
            FeatureSpec("skinfold", "Triceps skinfold", "mm", 7.0, 99.0, +1),
            FeatureSpec("pedigree", "Diabetes pedigree", "score", 0.08, 2.4, +1),
        ],
    ),
}


def get_task(key: str) -> TaskSpec:
    try:
        return TASKS[key]
    except KeyError as exc:
        raise KeyError(f"unknown task '{key}'. Available: {sorted(TASKS)}") from exc
