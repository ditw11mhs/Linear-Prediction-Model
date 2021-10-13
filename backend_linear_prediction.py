import numpy as np
import streamlit as st
import scipy.io.wavfile as wav
import os

@st.cache
def _load_data(data_path):
    load_data = np.loadtxt(data_path)
    raw_data_x = load_data[:, 0]
    raw_data_y = load_data[:, 1]
    padded_data_y = np.pad(raw_data_y, (len(raw_data_y), 0), "constant")
    return raw_data_x, raw_data_y, padded_data_y


@st.cache
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


class LinearModelPrediction:
    def __init__(self, data_path) -> None:
        self.raw_data_x, self.raw_data_y, self.padded_data_y = _load_data(data_path)

    def find_rxx_matrix(self, M):
        self.M = M
        auto_cor_matrix = _auto_correlation(
            M, raw_data_y=self.raw_data_y, padded_data_y=self.padded_data_y
        )
        # Mencari rxx
        self.rxx = auto_cor_matrix[: int(M) + 1, 0]

        # Mencari matriks Rxx-------

        #  Membuat rxx(0->P-1) terpadding
        padded_rxx = self.symetric_padding(self.rxx[:-1])
        rxx_matrix = np.lib.stride_tricks.sliding_window_view(
            padded_rxx, len(self.rxx[:-1])
        )
        rxx_matrix = np.flip(rxx_matrix, axis=0)
        self.rxx_matrix = rxx_matrix

    def symetric_padding(self, data):
        return np.pad(data, (len(data) - 1, 0), "reflect")

    def find_a_coef(self):

        # Membuat inverse matriks rxx
        self.inv_rxx_matrix = np.linalg.inv(self.rxx_matrix)

        self.a_coef = np.dot(self.inv_rxx_matrix, self.rxx[1:].reshape(-1, 1))

    def find_e_m(self):
        # Membuat matrix a inverse transpose
        a_inv_t = -self.a_coef.flatten()
        a_inv_t = np.insert(a_inv_t, 0, 1, axis=0)
        self.a_inv_t = a_inv_t
        # Membuat matriks agar dapat dikali dengan a invers transpose
        # bentuk akhir
        # x(0) x(1) x(2)
        # x(-1) x(0) x(1)
        # X(-2) x(-1) x(0)

        x = self.x_matrix_maker(a_inv_t)

        # Mencari error
        self.e_m = np.dot(a_inv_t.reshape(1, -1), x)

    def x_matrix_maker(self, a_inv_t, offset=0):
        x = np.lib.stride_tricks.sliding_window_view(
            self.padded_data_y[
                int(len(self.raw_data_y) - self.M) : int(
                    len(self.raw_data_y) * 2 - offset
                )
            ],
            len(a_inv_t),
        )

        x = np.flip(x, axis=1)
        return x.T

    def find_x_predict(self):
        a = self.a_coef.flatten()

        x = self.x_matrix_maker(a, offset=1)

        self.x_predict = np.dot(a.reshape(1, -1), x).flatten() + self.e_m

    def find_error_modelling(self):
        self.error_modelling = self.raw_data_y - self.x_predict

    def find_mse(self):
        self.mse = np.sum(np.square(self.error_modelling.flatten())) / (
            len(self.error_modelling.flatten()) - 1
        )

    def find_freq_resp(self):
        theta = np.arange(0, 2 * np.pi, 0.01).reshape(1, -1)

        real_cos_coef = np.arange(self.M + 1).reshape(-1, 1)
        real_cos_theta_coef = np.cos(theta * real_cos_coef)
        real = np.square(np.dot(self.a_inv_t, real_cos_theta_coef))

        imaj_cos_coef = np.arange(1, self.M + 1).reshape(-1, 1)
        imaj_cos_theta_coef = np.cos(theta * imaj_cos_coef)
        imaj = np.square(np.dot(self.a_coef.reshape(1, -1), imaj_cos_theta_coef))

        self.freq_resp = 1 / np.sqrt(real + imaj)
        self.theta = theta

    def process_all(self, M):
        self.find_rxx_matrix(M)
        self.find_a_coef()
        self.find_e_m()
        self.find_x_predict()
        self.find_error_modelling()
        self.find_mse()

    def to_sound(self):
        raw_data= self.raw_data_y.flatten()
        predicted_data = self.x_predict.flatten()
        min_r_d = np.min(raw_data)
        max_r_d =np.max(raw_data)
        min_p_d = np.min(predicted_data)
        max_p_d = np.max(predicted_data)
        new_max = 2147483647
        new_min = -2147483648
        
        norm_raw_data=(raw_data-min_r_d)/(max_r_d-min_r_d)*(new_max-new_min)+new_min
        norm_predicted_data=(predicted_data-min_p_d)/(max_p_d-min_p_d)*(new_max-new_min)+new_min
        
        wav.write(os.path.join("Data","raw_data.wav"), 3000, norm_raw_data.astype(np.int32))
        wav.write(os.path.join("Data","predicted_data.wav"), 3000, norm_predicted_data.astype(np.int32))