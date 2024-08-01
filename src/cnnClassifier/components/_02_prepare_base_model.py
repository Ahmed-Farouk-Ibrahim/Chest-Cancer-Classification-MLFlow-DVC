import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    """
    PrepareBaseModel class is responsible for loading a pre-trained base model,
    updating it with additional layers, and saving the model to specified paths.
    """

    def __init__(self, config: PrepareBaseModelConfig):
        """
        Initialize the PrepareBaseModel with the given configuration.

        Args:
            config (PrepareBaseModelConfig): Configuration object for preparing the base model.
        """
        self.config = config

    def get_base_model(self):
        """
        Load the pre-trained ResNet50 base model with the specified parameters and save it.
        """
        # Load the pre-trained VGG16 model with specified parameters
        self.model = tf.keras.applications.ResNet50(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        # Save the loaded base model to the specified path
        self.save_model(path=self.config.base_model_path, model=self.model)

    # @staticmethod decorator is used for methods that do not need to access the instance attributes (self) or class variables.
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        """
        Prepare the full model by adding custom layers to the base model and compiling it.

        Args:
            model (tf.keras.Model): The base model to which custom layers will be added.
            classes (int): Number of output classes for the final layer.
            freeze_all (bool): If True, freeze all layers in the base model.
            freeze_till (int or None): If specified, freeze layers up to this index.
            learning_rate (float): Learning rate for the optimizer.

        Returns:
            tf.keras.Model: The full model with added custom layers.
        """
        # Freeze layers in the base model as specified
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False

        # Add custom layers to the base model
        flatten_in = tf.keras.layers.GlobalAveragePooling2D()(model.output)
        dropout = tf.keras.layers.Dropout(0.35)(flatten_in)  # Adding dropout to prevent overfitting
        prediction = tf.keras.layers.Dense(units=classes, activation="softmax")(dropout)


        # Create the full model
        full_model = tf.keras.models.Model(inputs=model.input, outputs=prediction)


        # Compile the full model with specified optimizer, loss function, and metrics
        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),  # Changed optimizer to Adam
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        # Print the model summary
        full_model.summary()
        return full_model

    def update_base_model(self):
        """
        Update the base model by adding custom layers and compile it.
        Save the updated model to the specified path.
        """
        # Prepare the full model with custom layers and specified parameters
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=False,
            freeze_till=35,  # Fine-tune the last 35 layers
            learning_rate=self.config.params_learning_rate
        )

        # Save the updated full model to the specified path
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Save the given model to the specified path.

        Args:
            path (Path): Path to save the model.
            model (tf.keras.Model): The model to be saved.
        """
        model.save(path)
