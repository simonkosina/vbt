import numpy as np

class KalmanFilter(object):
    def __init__(self, x, y, dt, ux, uy, std_acc, xm_std, ym_std):
        """
        x: initial x position
        y: initial y position
        dt: sampling time
        ux: accelaration in x-direction
        uy: accelaration in y-direction
        std_acc: magnitude of the std of acceleration (process noise magnitude)
        xm_std: std of the measurement in x-direction
        ym_std: std of the measurement in y-direction
        """

        # Sampling time
        self.dt = dt
        
        # Control input variables
        self.u = np.array([[ux], [uy]])

        # Initial state
        self.x = np.array([[x], [y], [0], [0]])

        # State transition matrix
        self.A = np.array([[1, 0, self.dt, 0],
                             [0, 1, 0, self.dt],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])

        # Control input matrix
        self.B = np.array([[(self.dt**2)/2, 0],
                             [0, (self.dt**2)/2],
                             [self.dt, 0],
                             [0, self.dt]])
        
        # Measurement mapping matrix
        self.H = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0]])

        # Initial process noise covarience
        self.Q = np.array([[(self.dt**4)/4, 0, (self.dt**3)/2, 0],
                             [0, (self.dt**4)/4, 0, (self.dt**3)/2],
                             [(self.dt**3)/2, 0, self.dt**2, 0],
                             [0, (self.dt**3)/2, 0, self.dt**2]]) * std_acc**2 

        # Initial measurement noise covariance matrix
        self.R = np.array([[xm_std**2, 0],
                             [0, ym_std**2]])
        
        # Initial covariance matrix
        self.P = np.eye(self.A.shape[1])

    def predict(self):
        """Predict the state estimate x and error covariance P."""
        self.x = self.A @ self.x + self.B @ self.u
        self.P = self.A @ self.P @ self.A.T + self.Q

        # Return the predicted position
        return self.x[:2]
    
    def update(self, z):
        """
        Compute the Kalmain gain K and update the predicted
        state estimate x and predicted error covariance P.
        """
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ (z - self.H @ self.x)

        I = np.eye(self.H.shape[1])

        self.P = (I - (K @ self.H)) @ self.P

        # Return the estimated position
        return self.x[:2]
