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
        data1, data2 = self.file_loader()
        self.data1_len, self.data2_len = self.length(data1, data2)
        data2_padded = self.padder(data2)

        st.title("Cross Correlation")
        st.caption("Aditya Wardianto 07311940000001 - Biomodelling")

        col1,col2 = st.columns(2)
        # Time Lag
     
        with st.echo():
            self.t_lag = st.slider(label="Time Lag", min_value=-(self.data2_len-1), max_value=self.data2_len -
                                1, value=0, help="Slider to shift Data 2 position horizontaly")
            data2_out = data2_padded[self.data2_len -
                                    self.t_lag:2*self.data2_len-self.t_lag]
        
            # Plotting Input
            chart_input = pd.DataFrame(np.hstack(
                (data1.reshape(-1, 1), data2_out.reshape(-1, 1))), columns=['Data 1', 'Data 2'])
            st.line_chart(chart_input)

        # Correlation
        correlation = self.correlate(data1, data2_padded)
      
        #Plotting Correlation
        chart_output = pd.DataFrame(correlation, columns=['Correlation'])
        st.line_chart(chart_output)

    @st.cache(allow_output_mutation=True)
    def file_loader(self):

        root = os.path.dirname(sys.argv[0])
        path1 = r'Data\test.txt'
        path2 = r'Data\test2.txt'

        full_path1 = os.path.join(root, path1)
        full_path2 = os.path.join(root, path2)

        data1 = np.loadtxt(full_path1)
        data2 = np.loadtxt(full_path2)

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


if __name__ == "__main__":
    mainClass = Main()
    mainClass.main()
