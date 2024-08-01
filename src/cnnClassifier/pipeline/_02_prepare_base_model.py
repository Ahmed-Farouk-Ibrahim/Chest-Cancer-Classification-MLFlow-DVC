from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components._02_prepare_base_model import PrepareBaseModel
from cnnClassifier.logging import logger


STAGE_NAME = "Prepare base model"


class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        # Initialize the ConfigurationManager to load configuration and parameter settings
        config = ConfigurationManager()
        
        # Retrieve the configuration settings for preparing the base model
        prepare_base_model_config = config.get_prepare_base_model_config()
        
        # Initialize the PrepareBaseModel process with the configuration settings
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        
        # Download the pre-trained base model (e.g., VGG16) with specified configurations
        prepare_base_model.get_base_model()
        
        # Update the base model by adding custom layers and compiling it with specified parameters
        prepare_base_model.update_base_model()


if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> {STAGE_NAME} stage has started <<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> {STAGE_NAME} stage has completed <<<<<<\n\nx==========x")
    except Exception as e:
        # Log the exception details and raise it
        logger.exception(e)
        raise e
    