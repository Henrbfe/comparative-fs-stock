from model import model_interface
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam



class NeuralNetwork(model_interface.ModelInterface):
    """Neural net class model
    """
    def __init__(self, input_shape, output_activation='linear', activation='tanh', train_batch_size=32, use_l1=False, use_l2=False, use_batch_norm=False,custom_name=None,train_x=None, train_y=None, val_x = None, val_y=None):
        """Initialization to create neural net

        Args:
            input_shape: size of inputs
            output_activation: Type of output activation function. Defaults to 'linear'.
            activation: Type of activation function in network Defaults to 'tanh'.
            train_batch_size: Defaults to 32.
            use_l1: Boolean to use L1_regression. Defaults to False.
            use_l2: Boolean to use L1_regression. Defaults to False.
            use_batch_norm: Whether or not to use batch normalization. Defaults to False.
            custom_name: Custom name to differentiate save locations. Defaults to None.
            train_x: training inputs. Defaults to None.
            train_y: training targets. Defaults to None.
            val_x: validation.inputs Defaults to None.
            val_y: validation targets. Defaults to None.
        """
        self.optimizer = Adam(learning_rate=0.01)
        self.model = self._build_model(input_shape, output_activation, activation, use_l1, use_l2, use_batch_norm)
        self.model.compile(optimizer=self.optimizer, loss='mape')
        self.train_batch_size = train_batch_size
        self.custom_name=custom_name
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y
        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            
    def _get_name(self):
        return "NeuralNetwork" 
    
    def _get_custom_name(self):
        if self.custom_name is None:
            return self._get_name()
        else:
            return self.custom_name
        
    def _build_model(self,input_shape, output_activation='linear', activation='tanh', use_l1=False, use_l2=False, use_batch_norm=False):
        model = Sequential([
            Dropout(0.01, input_shape=input_shape),
            Dense(64, activation=activation, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=float(use_l1)/100, l2=float(use_l2)/100) if use_l1 or use_l2 else None),
            Dropout(0.4),
            Dense(64, activation=activation, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=float(use_l1)/100, l2=float(use_l2)/100) if use_l1 or use_l2 else None),
            Dropout(0.4),
            Dense(32, activation=activation, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=float(use_l1)/100, l2=float(use_l2)/100) if use_l1 or use_l2 else None),
            Dropout(0.3),
            Dense(32, activation=activation, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=float(use_l1)/100, l2=float(use_l2)/100) if use_l1 or use_l2 else None),
            Dropout(0.3),
            Dense(16, activation=activation, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=float(use_l1)/100, l2=float(use_l2)/100) if use_l1 or use_l2 else None),
            Dropout(0.2),
            Dense(16, activation=activation, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=float(use_l1)/100, l2=float(use_l2)/100) if use_l1 or use_l2 else None),
            Dropout(0.2),
            Dense(16, activation=activation, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=float(use_l1)/100, l2=float(use_l2)/100) if use_l1 or use_l2 else None),
            Dropout(0.1),
            Dense(1, activation=output_activation)
        ])

        # Compile the model with appropriate optimizer, loss, and metrics
        model.compile(optimizer=self.optimizer, loss='mape')

        if use_batch_norm:
            model = self.add_batch_normalization(model)
        
        return model

    def forward(self, X):
        """Method for predicting outputs

        Args:
            X: Input data

        Returns:
            np.array: predictions
        """
        return self.model.predict(X)

    def train(self, train_x, train_y, val_x, val_y, epochs=1000):
        if self.train_x is not None: 
            train_x = self.train_x
            train_y = self.train_y
            val_x = self.val_x
            val_y = self.val_y
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.model.fit(train_x, train_y, epochs=epochs, batch_size=self.train_batch_size, validation_data=(val_x, val_y), validation_freq=1, callbacks=[early_stopping])

    def save(self, file_path):
        """Method for saving graph

        Args:
            file_path: Save location
        """
        self.model.save(file_path + "nn_model.h5")
        
    def compile(self, loss, metrics,optimizer, *args, **kwargs):
        """Method for compiling graph

        Args:
            loss: type of loss, e.g. "mape"
            metrics: metrics
            optimizer: optimizer used
        """
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics, *args, **kwargs)


    @staticmethod
    def add_batch_normalization(model):
        new_model = Sequential()
        for layer in model.layers:
            new_model.add(layer)
            if isinstance(layer, Dense):
                new_model.add(BatchNormalization())
        return new_model

# Method outside the class for loading the model again
def load_model(file_path):
    """Method for loading saved file

    Args:
        file_path: file path

    Returns:
        model: neural net
    """
    return tf.keras.models.load_model(file_path)


def custom_tanh(x, new_min, new_max):
    """Custom tanh function that can be used as a scoring function

    Args:
        x: input value
        new_min: minmum value for tan_h function
        new_max: maximum value for tan_h function

    Returns:
        _type_: _description_
    """
    scaled = (x +1) / 2 #range [0,1]
    transformed = scaled * (new_max - new_min) + new_min
    return transformed




# improvements: #series: https://medium.com/@2020machinelearning/part-1-neural-network-regression-with-keras-and-tensorflow-introduction-and-default-dense-layers-39752d4fc05c
# overfitting: 
#   dropout layers
#   Adding randomness during training:
#       AdditiveGaussianNoise
#       RandomTermsCallback 
#   data augmentation: 
#
#   Avoid exploding gradient / no convergence
#       Reduce learning rate gradually
#   