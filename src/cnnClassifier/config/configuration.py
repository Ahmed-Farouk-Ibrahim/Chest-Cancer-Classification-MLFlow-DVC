import os
from src.cnnClassifier.constants import *
from src.cnnClassifier.utils.common import read_yaml, create_directories
from src.cnnClassifier.entity.config_entity import (DataIngestionConfig, PrepareBaseModelConfig,)

class ConfigurationManager:
    """
    ConfigurationManager handles the setup and retrieval of configuration settings for the pipeline.
    """

    def __init__(self, 
                 config_filepath=CONFIG_FILE_PATH, 
                 params_filepath=PARAMS_FILE_PATH):
        # Load configuration settings from the specified config.yaml file
        self.config = read_yaml(config_filepath)
        
        # Load parameters from the specified params file
        self.params = read_yaml(params_filepath)

        # Create the root directory for artifacts as declared in config.yaml file
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Retrieves and prepares the data ingestion configuration settings.

        Returns:
            DataIngestionConfig: Data ingestion configuration settings.
        """
        # Extract data_ingestion settings from config.yaml file
        config_di = self.config.data_ingestion  

        # Create directory for data ingestion artifacts
        create_directories([config_di.root_dir])

        # Create a DataIngestionConfig object with the settings
        data_ingestion_config = DataIngestionConfig(
            root_dir=config_di.root_dir,
            source_URL=config_di.source_URL,
            local_data_file=config_di.local_data_file,
            unzip_dir=config_di.unzip_dir
        )

        return data_ingestion_config


    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        """
        Retrieve the configuration for preparing the base model.

        Returns:
            PrepareBaseModelConfig: An object containing the configuration settings for preparing the base model.
        """
        # Extract the base model preparation configuration from the loaded config
        config = self.config.prepare_base_model

        # Create directories specified in the base model configuration
        create_directories([config.root_dir])

        # Instantiate and return the PrepareBaseModelConfig object with the necessary settings
        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config
    
    