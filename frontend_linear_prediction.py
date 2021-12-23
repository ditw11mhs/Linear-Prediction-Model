# from pandas.core.frame import DataFrame
import streamlit as st
from pandas import DataFrame as df
import plotly.express as px
import numpy as np
import os
from backend_linear_prediction import AudioWrapper, LinearModelPrediction


class Main:
    def __init__(self) -> None:
        pass

    def main(self) -> None:
        st.set_page_config(
            page_title="Tugas Linear Prediction Model Biomodelling ITS",
            page_icon=":chart_with_upwards_trend:",
            layout="wide",
        )
        with st.form("Recording"):
            c1,c2,_,_ = st.columns(4)
            duration = c1.number_input("Duration",1,100,3)
            rate = c2.number_input("Sample Rate",1,100000,3000)
            if st.form_submit_button("record"):
                audio =AudioWrapper(duration,rate)
                audio.record()

        self.input()
        if st.session_state["run"]==True:
            self.output()

    def input(self) -> None:
        if "run" not in st.session_state:
            st.session_state["run"] = False

        with st.sidebar.form("Input"):
            st.number_input(
                "Time Lag", key="time_lag", value=2, min_value=1, max_value=2000
            )
            st.checkbox(
                "No Big Data Frame",
                help="Better Rendering Performance On Big Time Lag",
                value=False,
                key="no_df",
            )
            if st.form_submit_button("Run"):
                st.session_state["run"] = True
                

    def output(self) -> None:
        st.title("Linear Prediction Model")
        # Class Initialization
        data_path = "recorded.txt"
        # data_path = os.path.join("Data", "Print_13_v2_PCG_RV.txt")
        self.linear_model = LinearModelPrediction(data_path)

        self.color = [
            "#99c9a8",
            "#cee7c1",
            "#a6d9c9",
            "#53aaba",
            "#2a85a3",
            "#7cd0d2",
            "#31d2d8",
        ]
        # Raw Data Plot
        st.header("Raw Signal")
        self.plot_raw_data()

        if st.session_state["run"]:
            self.style_format = "{:.5f}"
            self.dtype = np.float64
            # Find rxx and RXX
            self.find_rxx_Rxx()

            # Find a coefficient
            self.find_a_coef()

            #  Find e(m) (error)
            self.find_e_m()

            # Plot Raw Data with e(m)
            self.plot_raw_em()

            # Predict New data
            self.find_x_predict()

            # Find Modelling Error and MSE
            self.find_err_mod_mse()

            # Freq Resp
            self.find_freq_resp()
            # Compare Raw and Predicted Signal
            self.compare()

    def find_freq_resp(self):
        self.linear_model.find_freq_resp()
        st.subheader("Frequency Response")
        freq_resp_df = df(
            {
                "Frequency (Ohm)": self.linear_model.theta.flatten(),
                "H(z)": self.linear_model.freq_resp.flatten(),
                "A(z)": self.linear_model.a.flatten(),
            },
            dtype=self.dtype,
        )
        freq_resp_fig = px.line(
            freq_resp_df,
            x="Frequency (Ohm)",
            y=["H(z)", "A(z)"],
            color_discrete_sequence=[self.color[5], self.color[6]],
        )
        freq_resp_fig.update_layout(yaxis_title="Magnitude (dB)")
        st.plotly_chart(freq_resp_fig, use_container_width=True)

    def compare(self):
        self.linear_model.to_sound()
        st.subheader("Raw and Predicted Signal Comparison")
        compare_df = df(
            {
                "Raw Signal": self.linear_model.raw_data_y,
                "Predicted Signal": self.linear_model.x_predict.flatten(),
            },
            dtype=self.dtype,
        )
        compare_fig = px.line(
            compare_df,
            x="Predicted Signal",
            y="Raw Signal",
            color_discrete_sequence=[self.color[4]],
        )

        st.plotly_chart(compare_fig, use_container_width=True)
        st.plotly_chart(self.raw_data_fig_func(), use_container_width=True)
        st.audio(os.path.join("Data", "raw_data.wav"))
        st.plotly_chart(self.predict_data_fig, use_container_width=True)
        st.audio(os.path.join("Data", "predicted_data.wav"))

    def find_err_mod_mse(self):
        self.linear_model.find_error_modelling()
        self.linear_model.find_mse()

        st.subheader("Modeling Error")
        st.markdown(f"##### MSE: {self.linear_model.mse}")
        err_mod_df = df(
            {
                "t (second)": self.linear_model.raw_data_x,
                "Modelling Error": self.linear_model.error_modelling.flatten(),
            },
            # dtype=self.dtype,
        )
        err_mod_fig = px.line(
            err_mod_df,
            x="t (second)",
            y=["Modelling Error"],
            color_discrete_sequence=[self.color[3]],
        )
        err_mod_fig.update_layout(yaxis_title="V (mv)")
        st.plotly_chart(err_mod_fig, use_container_width=True)

    def find_x_predict(self):
        self.linear_model.find_x_predict()
        st.subheader("Predicted Signal")
        predict_data_df = df(
            {
                "t (second)": self.linear_model.raw_data_x,
                "Predicted Data": self.linear_model.x_predict.flatten(),
            },
            dtype=self.dtype,
        )
        predict_data_fig = px.line(
            predict_data_df,
            x="t (second)",
            y=["Predicted Data"],
            color_discrete_sequence=[self.color[2]],
        )
        predict_data_fig.update_layout(yaxis_title="V (mv)")
        self.predict_data_fig = predict_data_fig
        st.plotly_chart(predict_data_fig, use_container_width=True)

    @st.cache(max_entries=10, ttl=600)
    def raw_data_fig_func(self):
        raw_data_df = df(
            {
                "t (second)": self.linear_model.raw_data_x.flatten(),
                "Raw Data": self.linear_model.raw_data_y.flatten(),
            },
            # dtype=self.dtype,
        )
        raw_data_fig = px.line(
            raw_data_df,
            x="t (second)",
            y=[raw_data_df.columns[1]],
            color_discrete_sequence=[self.color[0]],
        )
        raw_data_fig.update_layout(yaxis_title="V (mv)")
        return raw_data_fig

    def plot_raw_data(self):
        st.plotly_chart(self.raw_data_fig_func(), use_container_width=True)

    def plot_raw_em(self):
        st.subheader("Raw Data & e(m) Plot")
        raw_data_em_df = df(
            {
                "t (second)": self.linear_model.raw_data_x,
                "Raw Data": self.linear_model.raw_data_y.flatten(),
                "e(m)": self.linear_model.e_m.flatten(),
            },
            dtype=self.dtype,
        )
        raw_data_em_fig = px.line(
            raw_data_em_df,
            x="t (second)",
            y=["Raw Data", "e(m)"],
            color_discrete_sequence=self.color[:2],
        )
        raw_data_em_fig.update_layout(yaxis_title="V (mv)")
        st.plotly_chart(raw_data_em_fig, use_container_width=True)

    def find_e_m(self):
        self.linear_model.find_e_m()

        st.subheader("e(m)")
        em_df = df(
            self.linear_model.e_m,
            dtype=self.dtype,
        )
        st.dataframe(em_df.style.format(self.style_format))

    def find_a_coef(
        self,
    ):
        self.linear_model.find_a_coef()
        cols_2 = st.columns(2)
        with cols_2[0]:
            st.subheader("Invers Rxx")
            if not st.session_state["no_df"]:
                inv_Rxx_df = df(
                    self.linear_model.inv_rxx_matrix.astype(np.float32),
                    index=np.arange(self.linear_model.inv_rxx_matrix.shape[0]).astype(
                        np.uint16
                    ),
                    columns=np.arange(self.linear_model.inv_rxx_matrix.shape[1]).astype(
                        np.uint16
                    ),
                    dtype=self.dtype,
                )
                st.dataframe(inv_Rxx_df.style.format(self.style_format))
            else:
                st.write("Hidden For Better Rendering Performance")

        with cols_2[1]:
            st.subheader("a Coefficient")
            a_coef_df = df(
                self.linear_model.a_coef,
                index=np.arange(len(self.linear_model.a_coef.flatten())).astype(
                    np.uint16
                ),
                columns=["a"],
                dtype=self.dtype,
            )
            st.dataframe(a_coef_df.style.format(self.style_format))

    def find_rxx_Rxx(self):
        self.linear_model.find_rxx_matrix(st.session_state["time_lag"])

        cols_1 = st.columns(2)
        with cols_1[0]:
            st.subheader("rxx")
            rxx_df = df(
                self.linear_model.rxx.flatten().astype(np.float32),
                columns=["rxx"],
                index=np.arange(len(self.linear_model.rxx.flatten())).astype(np.uint16),
                dtype=self.dtype,
            )
            st.dataframe(rxx_df.style.format(self.style_format))
        with cols_1[1]:
            st.subheader("Rxx")
            if not st.session_state["no_df"]:
                Rxx_df = df(
                    self.linear_model.rxx_matrix.astype(np.float32),
                    index=np.arange(self.linear_model.rxx_matrix.shape[0]).astype(
                        np.uint16
                    ),
                    columns=np.arange(self.linear_model.rxx_matrix.shape[1]).astype(
                        np.uint16
                    ),
                    dtype=self.dtype,
                )
                st.dataframe(Rxx_df)
            else:
                st.write("Hidden For Better Rendering Performance")


if __name__ == "__main__":
    main = Main()
    main.main()
    
    
