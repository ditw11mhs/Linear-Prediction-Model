import streamlit as st
import numpy as np
import pandas as pd
import os
import sys


class Main:
    def __init__(self):
        pass

    def main(self):
        # File Loading and Padding
        # st.set_page_config(layout="wide")

        # col1, col2 = st.columns(2)
        # with col1:
        data1, data2 = self.file_loader()
        self.data1_len, self.data2_len = self.length(data1, data2)
        data2_padded = self.padder(data2)

        st.title("Cross Correlation")
        st.caption("Aditya Wardianto 07311940000001 - Biomodelling")

        # Time Lag
        st.header("Time Lag")
        self.t_lag = st.slider(label="Time Lag", min_value=-(self.data2_len-1), max_value=self.data2_len -
                                1, value=0, help="Slider to shift Data 2 position horizontaly")
        data2_out = data2_padded[self.data2_len -
                                    self.t_lag:2*self.data2_len-self.t_lag]

        # Plotting Input
        st.header("Input Plot")
        chart_input = pd.DataFrame(np.hstack(
            (data1.reshape(-1, 1), data2_out.reshape(-1, 1))), columns=['Data 1', 'Data 2'])
        st.line_chart(chart_input)

        # Correlation
        correlation = self.correlate(data1, data2_padded)

        # Plotting Correlation
        st.header("Correlation Plot")
        chart_output = pd.DataFrame(correlation, columns=['Correlation'])
        st.line_chart(chart_output)

        # Normalization
        norm_correlation = self.normalize(correlation)

        # Plotting Normalized Correlation
        st.header("Normalization")
        chart_norm = pd.DataFrame(norm_correlation, columns=[
                                    'Normalized Correlation'])
        st.line_chart(chart_norm)

        # # with col2:
        #     st.write(pd.DataFrame(np.hstack((data1.reshape(-1, 1), data2_out.reshape(-1,
        #              1), correlation.reshape(-1,1)))), columns=['Data 1', 'Data 2', 'Correlation'])

    @st.cache(allow_output_mutation=True)
    def file_loader(self):

        # Deployment
        path1 = r'/app/Data/test.txt'
        path2 = r'/app/Data/test2.txt'

        # Local
        # path1 = r'Data\test.txt'
        # path2 = r'Data\test2.txt'

        data1 = np.loadtxt(path1)
        data2 = np.loadtxt(path2)

        return data1, data2

    @st.cache
    def length(self, data1, data2):
        return len(data1), len(data2)

    @st.cache
    def padder(self, data2):
        data2_padded = np.pad(
            data2, (self.data2_len, 2*self.data2_len), 'constant', constant_values=0)
        return data2_padded

    def correlate(self, data1, data2_padded):
        start = self.data2_len - self.t_lag
        end = 2*self.data2_len-self.t_lag+self.data1_len-1

        data2_1d = data2_padded[start:end]

        data2_2d = np.lib.stride_tricks.sliding_window_view(
            data2_1d, self.data1_len)
        correlation = np.dot(data2_2d, data1)
        return correlation

    def normalize(self, data):
        return (data - np.mean(data))/np.std(data)


if __name__ == "__main__":
    mainClass = Main()
    mainClass.main()
