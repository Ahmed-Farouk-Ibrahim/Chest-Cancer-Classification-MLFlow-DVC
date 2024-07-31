from src.cnnClassifier.logging import logger
from src.cnnClassifier.pipeline._01_data_ingestion import DataIngestionTrainingPipeline

STAGE_NAME = "Data Ingestion stage"

try:
    logger.info(f">>>>>> {STAGE_NAME} has started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> {STAGE_NAME} has completed <<<<<<\n\nx==========x")
except Exception as e:
    # Log the exception details and raise it
    logger.exception(e)
    raise e
