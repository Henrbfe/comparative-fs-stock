import pickle
from sklearn.svm import LinearSVR
# from thundersvm import SVR as ThunderSVR
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from model.model_interface import ModelInterface


class StockSVR(ModelInterface):
    """Support Vector Regressor class for stock price predictions."""

    def __init__(
        self,
        kernel: str,
        gamma: float,
        epsilon: float,
        C: float,
        tol: float = 0.001,
        custom_name: str = None,
        use_thunder_svm = False
    ):
        """Initialization of SVR

        Args:
            kernel (str): Kernel hyperparamter for SVR.
            gamma (float): Parameter used by kernel in SVR. 
            epsilon (float): Hyperparamter for SVR describing error margin.
            C (float): Hyperparamter for SVR deciding level of regularization.
            tol (float, optional): Tolerance for stopping criteria.  Defaults to 0.001.
            custom_name (str, optional): _description_. Defaults to None.
            use_thunder_svm (bool, optional): _description_. Defaults to False.
        """
        super().__init__()
        self.custom_name = custom_name
        if use_thunder_svm:
            # For gpu accelaration
            self.model = None #ThunderSVR(kernel=kernel, gamma=gamma, epsilon=epsilon, C=C, tol=tol)
        else:
            self.model = make_pipeline(
                Nystroem(kernel=kernel, gamma=gamma, n_jobs=-1), LinearSVR(epsilon=epsilon, C=C, tol=tol)
            )

    def train(self, train_x, train_y, val_x=None, val_y=None):
        """Fits the nystroem SVR on the provided training data and labels."""
        self.model.fit(train_x, train_y)

    def forward(self, inputs):
        """Performs prediction on the given inputs."""
        return self.model.predict(inputs)

    def evaluate(self, X, y):
        """Evaluates model using buildt-in method"""
        return self.model.score(X, y)

    def _get_custom_name(self):
        if self.custom_name is None:
            return self._get_name()
        else:
            return self.custom_name

    def _get_name(self):
        return "SVR"

    def save(self, filepath: str):
        """Method for saving SVR"""
        pickle.dump(self.model, open(filepath + "/svr_model.pickle", "wb"))

    # def evaluate(self, X, y):
    #     return mean_absolute_percentage_error(y,self.forward(X))


# def load_SVR(filepath, epsilon: float, tol: float, C: float, custom_name: str):
#     loaded_model = pickle.load(open(filepath, "rb"))
#     svr = SVR(epsilon, tol, C, custom_name)
#     svr.model = loaded_model
#     return svr
