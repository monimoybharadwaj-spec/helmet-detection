from dataclasses import dataclass

# Model Trainer Artifact
@dataclass
class ModelTrainerArtifact:
    model_path: str
    metrics: dict

# Model Evaluation Artifact
@dataclass
class ModelEvaluationArtifact:
    mAP50: float
    precision: float
    recall: float