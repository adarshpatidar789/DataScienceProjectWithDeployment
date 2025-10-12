from src.datascience import logger
from src.datascience.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline

STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>>> stage {STAGE_NAME} started >>>>>")
    obj = DataIngestionTrainingPipeline()
    obj.initiate_date_ingestion()
    logger.info(f">>>>> stage {STAGE_NAME} completed >>>>>")
except Exception as e:
    logger.exception(e)
    raise e