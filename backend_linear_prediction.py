import numpy as np
import streamlit as st
import scipy.io.wavfile as wav
import os
from numba import jit, njit


@st.cache(max_entries=10, ttl=600)
def _load_data(data_path):
    load_data = np.loadtxt(data_path)
    raw_data_x = load_data[:, 0]
    raw_data_y = load_data[:, 1]
    padded_data_y = np.pad(raw_data_y, (len(raw_data_y), 0), "constant")
    return raw_data_x, raw_data_y, padded_data_y


@st.cache(max_entries=10, ttl=600)
def _auto_correlation(M, raw_data_y, padded_data_y):
    # Meyimpan data raw
    data = raw_data_y

    # Mendapatkan data dalam bentuk matriks kolom sebesar [N Data,1]
    data_m = data.reshape(-1, 1)

    # Mendapatkan matriks untuk melakukan auto korelasi [N Data, N Data]
    data_m_l = np.lib.stride_tricks.sliding_window_view(padded_data_y, len(data))
    data_m_l = np.flip(data_m_l, axis=0)
    auto_cor_matrix = np.dot(data_m_l, data_m)
    return auto_cor_matrix


@jit(forceobj=True, looplift=True)
def _get_rxx_matrix(rxx):
    padded_rxx = np.pad(rxx[:-1], (len(rxx[:-1]) - 1, 0), "reflect")
    rxx_matrix = np.lib.stride_tricks.sliding_window_view(padded_rxx, len(rxx[:-1]))
    rxx_matrix = np.flip(rxx_matrix, axis=0)
    return rxx_matrix


@jit(forceobj=True, looplift=True)
def _x_matrix_maker(padded_data_y, raw_data_y, M, a_inv_t, offset=0):
    x = np.lib.stride_tricks.sliding_window_view(
        padded_data_y[int(len(raw_data_y) - M) : int(len(raw_data_y) * 2 - offset)],
        len(a_inv_t),
    )
    x = np.flip(x, axis=1)
    return x.T


@njit
def _inverse_matrix(rxx_matrix):
    return np.linalg.inv(rxx_matrix)


@njit
def _get_e_m(a_inv_t, x):
    out = np.dot(a_inv_t.reshape(1, -1).astype(np.float64), x.astype(np.float64))
    return out


@njit
def _get_x_predict(a, x, e_m):
    return np.dot(
        a.reshape(1, -1).astype(np.float64), x.astype(np.float64)
    ).flatten() + e_m.astype(np.float64)


@njit
def _mse(error_modelling):
    return np.sum(np.square(error_modelling.flatten())) / (
        len(error_modelling.flatten()) - 1
    )


@njit
def _mag_theta(M, a_coef):
    theta = np.arange(0, np.pi, 0.01).reshape(1, -1)
    theta_coef = np.arange(1, M + 1).reshape(-1, 1) * theta
    min_j_theta_coef = -1j * theta_coef
    exp_min_j_theta_coef = np.exp(min_j_theta_coef)
    a_coef_dot = np.dot(
        a_coef.reshape(1, -1).astype(np.complex64),
        exp_min_j_theta_coef.astype(np.complex64),
    )
    H_z = 1 / (1 - a_coef_dot)
    mag_h_z = np.abs(H_z)
    return theta, mag_h_z


@njit
def _norm_to_wav(raw_data_y, x_predict):
    raw_data = raw_data_y.flatten()
    predicted_data = x_predict.flatten()
    min_r_d = np.min(raw_data)
    max_r_d = np.max(raw_data)
    min_p_d = np.min(predicted_data)
    max_p_d = np.max(predicted_data)
    new_max = 2147483647
    new_min = -2147483648

    norm_raw_data = (raw_data - min_r_d) / (max_r_d - min_r_d) * (
        new_max - new_min
    ) + new_min
    norm_predicted_data = (predicted_data - min_p_d) / (max_p_d - min_p_d) * (
        new_max - new_min
    ) + new_min

    return norm_raw_data, norm_predicted_data


class LinearModelPrediction:
    def __init__(self, data_path) -> None:
        self.raw_data_x, self.raw_data_y, self.padded_data_y = _load_data(data_path)

    def find_rxx_matrix(self, M):
        self.M = M

        # Mencari rxx
        self.rxx = _auto_correlation(
            M, raw_data_y=self.raw_data_y, padded_data_y=self.padded_data_y
        )[: int(M) + 1, 0]

        # Mencari matriks Rxx-
        self.rxx_matrix = _get_rxx_matrix(self.rxx)

    def find_a_coef(self):

        # Membuat inverse matriks rxx
        self.inv_rxx_matrix = _inverse_matrix(self.rxx_matrix)

        self.a_coef = np.dot(self.inv_rxx_matrix, self.rxx[1:].reshape(-1, 1)).flatten()

    def find_e_m(self):
        # Membuat matrix a inverse transpose
        self.a_inv_t = np.insert(-self.a_coef, 0, 1, axis=0)
        # Membuat matriks agar dapat dikali dengan a invers transpose
        # bentuk akhir
        # x(0) x(1) x(2)
        # x(-1) x(0) x(1)
        # X(-2) x(-1) x(0)

        x = _x_matrix_maker(self.padded_data_y, self.raw_data_y, self.M, self.a_inv_t)

        # Mencari error
        self.e_m = _get_e_m(self.a_inv_t, x)

    def find_x_predict(self):

        x = _x_matrix_maker(
            self.padded_data_y, self.raw_data_y, self.M, self.a_coef, offset=1
        )

        self.x_predict = _get_x_predict(self.a_coef, x, self.e_m)

    def find_error_modelling(self):
        self.error_modelling = self.raw_data_y - self.x_predict

    def find_mse(self):
        self.mse = _mse(self.error_modelling)

    def find_freq_resp(self):
        self.theta, self.freq_resp = _mag_theta(self.M, self.a_coef)

    def to_sound(self):
        norm_raw_data, norm_predicted_data = _norm_to_wav(
            self.raw_data_y, self.x_predict
        )

        wav.write(
            os.path.join("Data", "raw_data.wav"), 3000, norm_raw_data.astype(np.int32)
        )
        wav.write(
            os.path.join("Data", "predicted_data.wav"),
            3000,
            norm_predicted_data.astype(np.int32),
        )
