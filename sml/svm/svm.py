import jax
import jax.numpy as jnp

from sml.svm.smo import SMO

class SVM():
    """
    Parameters
    ----------
    kernel : str, default="rbf"
        The kernel function used in the svm algorithm, maps samples 
        to a higher dimensional feature space.

    C : float
        Error penalty coefficient.

    gamma : str, default="scale"
        The coefficient in the kernel function

    max_iter : int, default=300
        Maximum number of iterations of the svm algorithm for a
        single run.

    tol : float, default=1e-3
        Acceptable error to consider the two to be equal.
    """
    
    def __init__(self, kernel="rbf", C=1.0, gamma = 'scale', max_iter=102, tol=1e-3):
        
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.n_features = None

        self.alpha = None
        self.alph_y = None
        self.b = None
        
    def cal_kernel(self,x, x_):

        if type(self.gamma) == str:
            gamma = {
                'scale': 1 / (self.n_features * x.var()),
                'auto': 1 / self.n_features,
            }[self.gamma]
        else:
            gamma = self.gamma
        if(self.kernel=="rbf"):
            kernel_res = jnp.exp(-(1 / (self.n_features * x.var())) * ((x**2).sum(1, keepdims=True) + (x_**2).sum(1) - 2 * jnp.matmul(x, x_.T)))
        '''
        add other kernel
        '''

        return kernel_res
        
    def cal_Q(self,x,y):
        kernel_res = self.cal_kernel(x,x)
        Q = y.reshape(-1, 1) * y * kernel_res
        return Q
    
    
    def fit(self, X, y):
        """Fit SVM.

        Using the Sequential Minimal Optimization(SMO) algorithm to solve the Quadratic programming problem in
        the SVM, which decomposes the large optimization problem to several small optimization problems. Firstly, 
        the SMO algorithm select alpha_i and alpha_j by 'smo.working_set_select_i()' and 'smo.working_set_select_j'. 
        Secondly, the SMO algorithm update the parameter by 'smo.update()'. 


        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Input data.

        y : {array-like}, shape (n_samples)
            Lable of the input data.

        """

        l, self.n_features = X.shape
        p = -jnp.ones(l)
        smo = SMO(l,self.C, self.tol)
        tol=self.tol
        i0,j0 = -1,-1
        Q = self.cal_Q(X, y)
        alpha = 0.0 * y
        neg_y_grad = -p * y
        for n_iter in range(self.max_iter):
            i = smo.working_set_select_i(alpha, y, neg_y_grad)
            j = smo.working_set_select_j(i, alpha, y, neg_y_grad,Q)
            neg_y_grad, alpha  = smo.update(i, j, Q, y, alpha, neg_y_grad)

        self.alpha = alpha
        self.b = smo.calculate_b(alpha,neg_y_grad,y)
        self.alpha_y = self.alpha * y


    def predict(self, X, y, x):
        """Result estimates.

        Calculate the classification result of the input data.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Input data.

        y : {array-like}, shape (n_samples)
            Lable of the input data.
        
        x : {array-like}, shape (n_samples, n_features)
            Input data for prediction.

        Returns
        -------
        ndarray of shape (n_samples)
            Returns the classification result of the input data for prediction.
        """
        
        pred = jnp.matmul(
            self.alpha * y,
            self.cal_kernel(X, x),
        ) + self.b
        return (pred>=0).astype(int)

