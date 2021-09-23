import streamlit as st
import numpy as np
import pandas as pd
import os
import sys


class Main:
    def __init__(self):
        pass

    def main(self):
        self.deploy = True

        # Page Configuration

        if self.deploy:
            favicon_path = r"/app/Data/favicon-web.ico"
        else:
            favicon_path = r"Data\favicon-web.ico"

        st.set_page_config(
            page_title='Tugas Cross Correlation Biomodelling ITS', page_icon=favicon_path, layout="wide")

        # File Loading and Padding

        data1, data2 = self.file_loader()
        self.data1_len, self.data2_len = self.length(data1, data2)
        data2_padded = self.padder(data2)

        st.title("Cross Correlation")
        st.caption("Aditya Wardianto 07311940000001 - Biomodelling ITS")
        st.markdown("[Github Link](https://github.com/ditw11mhs/CrossCorrelation)")

        # Time Lag
        st.header("Time Lag")
        self.t_lag = st.slider(label="Time Lag", min_value=-(self.data2_len-1), max_value=self.data2_len -
                               1, value=0, help="Slider to shift Data 2 position horizontaly")
        data2_out = data2_padded[self.data2_len -
                                 self.t_lag:2*self.data2_len-self.t_lag]

        col1, col2 = st.columns(2)
        # Plotting Input
        with col1:
            st.header("Input Plot")
            chart_input = pd.DataFrame(
                {
                    "Data 1": data1,
                    "Data 2": data2
                }
            )
            st.line_chart(chart_input)

        # Correlation
        correlation = self.correlate(data1, data2_padded)
        

        col3, col4 = st.columns(2)
        with col3:
            # Plotting Correlation
            st.header("Correlation Plot")
            chart_output = pd.DataFrame({'Correlation': correlation})
            st.line_chart(chart_output)

        # Normalization
        norm_correlation = self.normalize(correlation)

        with col2:
            # Plotting Normalized Correlation
            st.header("Normalization")
            chart_norm = pd.DataFrame(
                {'Normalized Correlation': norm_correlation})
            st.line_chart(chart_norm)

        with col4:
            st.header('Data Table')
            st.write(pd.DataFrame({
                'Data 1': data1,
                'Data 2': data2_out,
                'Correlation': correlation,
                'Normalized Correlation': norm_correlation
            }
            ))

    @st.cache(allow_output_mutation=True)
    def file_loader(self):

        # if self.deploy:
        #     path1 = r'/app/Data/test.txt'
        #     path2 = r'/app/Data/test2.txt'
        # else:
        #     path1 = r'Data\test.txt'
        #     path2 = r'Data\test2.txt'

        # data1 = np.loadtxt(path1)
        # data2 = np.loadtxt(path2)
        x = np.linspace(0, 1, 10000)
        data1 = np.sin(5*np.pi*x)
        data2 = np.sin(5*np.pi*x)

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
        # return (data - np.mean(data))/np.std(data)
        return (data-np.min(data))/(np.max(data)-np.min(data))


if __name__ == "__main__":
    mainClass = Main()
    mainClass.main()
