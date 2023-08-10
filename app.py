import numpy as np
import os
import pandas as pd
import streamlit as st
import torch

from src.LSTM_model import Training_LSTM
from sklearn.preprocessing import MinMaxScaler, StandardScaler

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def main():

    st.write('## Importing the data')

    data_options = st.selectbox(label='New .csv file or an example?',
                                options=['New', 'Example'])
    if data_options == 'Example':
        csv_examples = ['Amazon', 'Electric Production', 'Passengers']
        uploaded_file = st.selectbox(label='Choose column with Values',
                                     options=csv_examples)
        st.write(uploaded_file)
        if uploaded_file == 'Amazon':
            csv_path = os.path.join('examples', 'amazon.csv')
            df = pd.read_csv(csv_path)
        elif uploaded_file == 'Electric Production':
            csv_path = os.path.join('examples', 'Electric_Production.csv')
            df = pd.read_csv(csv_path)
        else:
            csv_path = os.path.join('examples', 'passengers.csv')
            df = pd.read_csv(csv_path)
        data_uploaded = True

    elif data_options == 'New':
        data_uploaded = False
        uploaded_file = st.file_uploader('Import your file here', type='csv')
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            data_uploaded = True

    else:
        data_uploaded = False

    if data_uploaded is True:
        dates_col = st.selectbox(label='Choose column with Dates',
                                 options=df.columns)
        opt_cols = [val for val in df.columns if val != dates_col]
        vals_col = st.selectbox(label='Choose column with Values',
                                options=opt_cols)
        df = df[[dates_col, vals_col]]
        df[dates_col] = pd.to_datetime(df[dates_col])
        st.line_chart(data=df, x=dates_col, y=vals_col)

        st.write('## Selecting parameters')
        selected_scaler = st.selectbox(label='Choose method to transform data',
                                       options=('Standard', 'MinMax'))
        if selected_scaler == 'Standard':
            data_scaler = MinMaxScaler(feature_range=(-1, 1))
        else:
            data_scaler = StandardScaler()

        window_size = st.slider(label='Window size',
                                min_value=2,
                                max_value=10)

        split_size = st.slider(label='Split size',
                               min_value=0.7,
                               max_value=0.95)

        batch_size = st.selectbox(label='Batch size',
                                  options=(1, 2, 4, 8, 16, 32))

        hidden_size = st.slider(label='Hidden size',
                                min_value=1,
                                max_value=50)

        stacked_layers = st.slider(label='Num of stacked layers',
                                   min_value=1,
                                   max_value=4)

        mod = Training_LSTM(df,
                            window_size=window_size,
                            split_size=split_size,
                            batch_size=batch_size,
                            scaler=data_scaler,
                            dates_col=dates_col,
                            vals_col=vals_col,
                            input_size=1,
                            hidden_size=hidden_size,
                            num_stacked_layers=stacked_layers,
                            device=device)

        train_size = int(len(mod.X) * mod.split_size)

        if st.button(label='Train the model'):

            mod.training(verbose=False)

            train_pred = np.ones_like(df[vals_col]) * np.nan
            train_transform = mod.convert_vals(
                mod.X_train, mod.y_train, True)
            train_pred[window_size-1:train_size-1] = train_transform[
                :train_size-window_size]
            train_pred = pd.Series(train_pred)
            # train_transform = mod.convert_vals(
            #     mod.X_train, mod.y_train, True)[window_size:]
            # train_pred[
            #     (window_size*2-1):train_size+window_size-1] = train_transform

            test_pred = np.ones_like(df[vals_col]) * np.nan
            test_pred[train_size+window_size:] = mod.convert_vals(
                mod.X_test, mod.y_test, True)
            test_pred = pd.Series(test_pred)

            df_plot = pd.concat(
                [df[dates_col], df[vals_col], train_pred, test_pred], axis=1)
            df_plot.columns = ['Dates', 'Real_vals', 'Train_pred', 'Test_pred']
            st.line_chart(data=df_plot, x='Dates')

            st.write('## Predicting next value')
            st.write('Next value prediction is:')
            pred_val = mod.pred_vals()
            last_val = mod.last_series
            diff_vals = round(pred_val-last_val, 3)
            st.metric(label='Predicted Value',
                      value=round(pred_val, 3),
                      delta=diff_vals)
            st.balloons()


if __name__ == '__main__':
    main()
