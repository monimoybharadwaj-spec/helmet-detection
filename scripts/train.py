from src.helmet.pipelines.training_pipeline import ModelTrainingPipeline

if __name__ == "__main__":
    pipeline = ModelTrainingPipeline()

    artifact = pipeline.pipeline_run()

    print("Training Completed")
    print(f"Model saved at: {artifact.model_path}")
    print(f"Metrics: {artifact.metrics}")