import numpy as np
import scipy.linalg


"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        # 因为咱们的运动观测空间是4维还有每个维度的隐含的速度变量
        # （速度变量可以靠我们自己推算出来，但检测器所能够检测出来的信息只有这4个维度，
        # 故状态空间8，测量空间4）
        # （x, y, a, h, vx, vy, va, vh），
        # 所以建立一个8*8矩阵的状态空间，每个维度的（ndim+i）代表各自的速度
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)  # self._motion_mat状态转移矩阵。np.eye返回一个二维数组(N,M)，对角线的地方为1，其余的地方为0.
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)  # 状态转换矩阵

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20  # 位置的方差（人为设定）
        self._std_weight_velocity = 1. / 160  # 速度的方差（人为设定）

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)  # 产生一个和mean_pos一样大小的全0数组
        # 根据当前坐标初始化追踪目标的状态空间，并在初始时将速度假设为0
        mean = np.r_[mean_pos, mean_vel]  # np.r_是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等。

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        # 分别根据上面给定的std来形成8*8的协方差矩阵，
        # 至于为什么乘以measurement[3]，这是作者个人设定，咱们也可以修改他为measurement[2]等等
        covariance = np.diag(np.square(std))  # 返回矩阵的对角线元素或者返回一个二维数组
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        # 相当于得到t时刻估计值。 Q 预测过程中噪声协方差
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))  # np.r_ 按列连接两个矩阵，初始化噪声矩阵 Q

        # mean为目标当前状态，乘以转移矩阵得到预测状态
        mean = np.dot(self._motion_mat, mean)  # x' = Fx
        # 对协方差矩阵进行更新。np.linalg.multi_dot计算多个矩阵的乘法/点积
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov  # P' = FPF^T + Q

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        # R 测量过程中噪声的协方差
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))  # 初始化噪声矩阵R

        # 状态到观测的转换矩阵乘以测量值得到状态空间的值，即由8维状态空间转换到4维测量空间
        mean = np.dot(self._update_mat, mean)  # 将均值向量映射到检测空间，即Hx'
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))  # 将协方差矩阵映射到检测空间，即HP'H^T
        return mean, covariance + innovation_cov

    # 通过估计值和观测值估计最新结果
    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.  根据真实测量值对预测值进行修正

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        # 将均值和协方差映射到检测空间，得到 Hx' 和 S   S = HP'H^T + R，R是目标检测器的噪声矩阵，是一个4x4的对角矩阵。对角线上的值分别为中心点两个坐标以及宽高的噪声。
        projected_mean, projected_cov = self.project(mean, covariance)

        # 矩阵分解。解矩阵方程
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        # 计算卡尔曼增益K，是作用于衡量估计误差的权重。  K = P'H^TS^-1
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        # y = z - Hx'  z是Detection的mean，不包含变化值，状态为[cx,cy,a,h]。H是测量矩阵，将Track的均值向量x'映射到检测空间。计算的y是Detection和Track的均值误差。
        innovation = measurement - projected_mean

        # x = x' + Ky   更新后的均值向量x。
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        # P = (I - KH)P'  更新后的协方差矩阵。
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.

        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
