import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
from backend_linear_prediction import LinearModelPrediction


class Main:
    def __init__(self) -> None:
        pass

    def main(self) -> None:
        st.set_page_config(
            page_title="Tugas Linear Prediction Model Biomodelling ITS",
            page_icon=":chart_with_upwards_trend:",
            layout="wide",
        )

        self.input()

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
        data_path = os.path.join("Data", "Print_13_v2_PCG_RV.txt")
        linear_model = LinearModelPrediction(data_path)

        self.color = ["#8CDFD6", "#1E555C", "#F4D8CD", "#EDB183", "#F15152", "#FFFFFF"]
        # Raw Data Plot
        self.plot_raw_data(linear_model)

        if st.session_state["run"]:
            style_format = "{:.5f}"

            # Find rxx and RXX
            self.find_rxx_Rxx(linear_model, style_format)

            # Find a coefficient
            self.find_a_coef(linear_model, style_format)

            #  Find e(m) (error)
            self.find_e_m(linear_model, style_format)

            # Plot Raw Data with e(m)
            self.plot_raw_em(linear_model)

            # Predict New data
            self.find_x_predict(linear_model)

            # Find Modelling Error and MSE
            self.find_err_mod_mse(linear_model)

            # Freq Resp
            self.find_freq_resp(linear_model)
            # Compare Raw and Predicted Signal
            self.compare(linear_model)
            
            
            

    def find_freq_resp(self, linear_model):
        linear_model.find_freq_resp()
        st.subheader("Frequency Response")
        freq_resp_df = pd.DataFrame(
            {
                "Frequency (Ohm)": linear_model.theta.flatten(),
                "Magnitude (dB)": linear_model.freq_resp.flatten(),
            }
        )
        freq_resp_fig = px.line(
            freq_resp_df,
            x="Frequency (Ohm)",
            y=["Magnitude (dB)"],
            color_discrete_sequence=[self.color[5]],
        )
        freq_resp_fig.update_layout(yaxis_title="Magnitude (dB)")
        st.plotly_chart(freq_resp_fig, use_container_width=True)

    def compare(self, linear_model):
        linear_model.to_sound()
        st.subheader("Raw and Predicted Signal Comparison")
        compare_df = pd.DataFrame(
            {
                "Raw Signal": linear_model.raw_data_y,
                "Predicted Signal": linear_model.x_predict.flatten(),
            }
        )
        compare_fig = px.line(
            compare_df,
            x="Predicted Signal",
            y="Raw Signal",
            color_discrete_sequence=[self.color[4]],
        )

        st.plotly_chart(compare_fig, use_container_width=True)
        st.plotly_chart(self.raw_data_fig, use_container_width=True)
        st.audio(os.path.join("Data","raw_data.wav"))
        st.plotly_chart(self.predict_data_fig, use_container_width=True)
        st.audio(os.path.join("Data","predicted_data.wav"))

    def find_err_mod_mse(self, linear_model):
        linear_model.find_error_modelling()
        linear_model.find_mse()

        st.subheader("Modeling Error")
        st.markdown(f"##### MSE: {linear_model.mse}")
        err_mod_df = pd.DataFrame(
            {
                "t (second)": linear_model.raw_data_x,
                "Modelling Error": linear_model.error_modelling.flatten(),
            }
        )
        err_mod_fig = px.line(
            err_mod_df,
            x="t (second)",
            y=["Modelling Error"],
            color_discrete_sequence=[self.color[3]],
        )
        err_mod_fig.update_layout(yaxis_title="V (mv)")
        st.plotly_chart(err_mod_fig, use_container_width=True)

    def find_x_predict(self, linear_model):
        linear_model.find_x_predict()
        st.subheader("Predicted Signal")
        predict_data_df = pd.DataFrame(
            {
                "t (second)": linear_model.raw_data_x,
                "Predicted Data": linear_model.x_predict.flatten(),
            }
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

    def plot_raw_data(self, linear_model):
        st.header("Raw Signal")
        raw_data_df = pd.DataFrame(
            {"t (second)": linear_model.raw_data_x, "Raw Data": linear_model.raw_data_y}
        )
        raw_data_fig = px.line(
            raw_data_df,
            x="t (second)",
            y=[raw_data_df.columns[1]],
            color_discrete_sequence=[self.color[0]],
        )
        raw_data_fig.update_layout(yaxis_title="V (mv)")
        self.raw_data_fig = raw_data_fig
        st.plotly_chart(raw_data_fig, use_container_width=True)

    def plot_raw_em(self, linear_model):
        st.subheader("Raw Data & e(m) Plot")
        raw_data_em_df = pd.DataFrame(
            {
                "t (second)": linear_model.raw_data_x,
                "Raw Data": linear_model.raw_data_y.flatten(),
                "e(m)": linear_model.e_m.flatten(),
            }
        )
        raw_data_em_fig = px.line(
            raw_data_em_df,
            x="t (second)",
            y=["Raw Data", "e(m)"],
            color_discrete_sequence=self.color[:2],
        )
        raw_data_em_fig.update_layout(yaxis_title="V (mv)")
        st.plotly_chart(raw_data_em_fig, use_container_width=True)

    def find_e_m(self, linear_model, style_format):
        linear_model.find_e_m()
        cols_3 = st.columns(2)
        # with cols_3[0]:
        st.subheader("e(m)")
        em_df = pd.DataFrame(linear_model.e_m, dtype=np.float16)
        st.dataframe(em_df.style.format(style_format))

    def find_a_coef(self, linear_model, style_format):
        linear_model.find_a_coef()
        cols_2 = st.columns(2)
        with cols_2[0]:
            st.subheader("Invers Rxx")
            if not st.session_state["no_df"]:
                inv_Rxx_df = pd.DataFrame(
                    linear_model.inv_rxx_matrix,
                    index=np.arange(linear_model.inv_rxx_matrix.shape[0]).astype(str),
                    columns=np.arange(linear_model.inv_rxx_matrix.shape[1]).astype(str),
                    dtype=np.float16,
                )
                st.dataframe(inv_Rxx_df.style.format(style_format))
            else:
                st.write("Hidden For Better Rendering Performance")

        with cols_2[1]:
            st.subheader("a Coefficient")
            a_coef_df = pd.DataFrame(
                linear_model.a_coef,
                index=[str(i) for i in np.arange(1, linear_model.a_coef.shape[0] + 1)],
                columns=["a"],
                dtype=np.float16,
            )
            st.dataframe(a_coef_df.style.format(style_format))

    def find_rxx_Rxx(self, linear_model, style_format):
        linear_model.find_rxx_matrix(st.session_state["time_lag"])

        cols_1 = st.columns(2)
        with cols_1[0]:
            st.subheader("rxx")
            rxx_df = pd.DataFrame({"rxx": linear_model.rxx.flatten()})
            st.dataframe(rxx_df.style.format(style_format))

        with cols_1[1]:
            st.subheader("Rxx")
            if not st.session_state["no_df"]:
                Rxx_df = pd.DataFrame(
                    linear_model.rxx_matrix,
                    index=np.arange(linear_model.rxx_matrix.shape[0]).astype(str),
                    columns=np.arange(linear_model.rxx_matrix.shape[1]).astype(str),
                    dtype=np.float16,
                )
                st.dataframe(Rxx_df.style.format(style_format))
            else:
                st.write("Hidden For Better Rendering Performance")


if __name__ == "__main__":
    main = Main()
    main.main()
