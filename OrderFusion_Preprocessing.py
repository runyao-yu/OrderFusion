import pandas as pd
import numpy as np
import os
import joblib
import warnings
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
warnings.filterwarnings("ignore")


'''
-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
-- -- -- -- -- -- functions used for filtering matched trades  -- -- -- -- -- --
-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
'''


def add_traded_volume(df):

    # Sort by OrderId then TransactionTime
    df = df.sort_values(['OrderId', 'TransactionTime'])
    
    # Identify trades
    trades_mask = df['ActionCode'].isin(['P', 'M'])
    
    # Compute the volume difference by shifting within each OrderId
    df['VolumeTraded'] = df.groupby('OrderId')['Volume'].shift(1) - df['Volume']
    
    # For the first row of each OrderId, set VolumeTraded to NaN
    is_first_in_group = df['OrderId'].ne(df['OrderId'].shift(1))
    df.loc[is_first_in_group, 'VolumeTraded'] = float('nan')
    
    # Set traded volume to 0 for non-trade rows
    df.loc[~trades_mask, 'VolumeTraded'] = 0
    
    # Drop rows where VolumeTraded is 0 or NaN
    df = df[df['VolumeTraded'].notna() & (df['VolumeTraded'] != 0)]
    
    return df

def filter_raw_data(country, year):
    base_path = f"EPEX_Spot_Orderbook/{country}/Intraday Continuous/Orders"
    
    # Only load these columns from CSV
    necessary_columns = [
        'DeliveryStart',
        'Side',
        'Product',
        'Price',
        'Volume',
        'ActionCode',
        'TransactionTime',
        'OrderId'
    ]
    
    path = os.path.join(base_path, str(year))
    
    # We will collect results in lists and concatenate once
    hour_list = []
    quarter_hour_list = []
    
    # Count the number of files for tqdm progress bar
    total_files = sum(len(files) for _, _, files in os.walk(path))
    
    with tqdm(total=total_files, desc=f"processing {year} data for {country}", unit="file") as pbar:
        for dirname, _, filenames in os.walk(path):
            for filename in filenames:
                data_path = os.path.join(dirname, filename)
                
                # Read CSV with only necessary columns
                df = pd.read_csv(
                    data_path,
                    header=1,
                    dtype={'ParentId': 'Int64'},
                    usecols=necessary_columns
                )
                
                # Split into hour and quarter-hour subsets
                hour_df = df[df['Product'].isin(['Intraday_Hour_Power', 'XBID_Hour_Power'])]
                qh_df = df[df['Product'].isin(['Intraday_Quarter_Hour_Power', 'XBID_Quarter_Hour_Power'])]
                
                # Process the hour trades
                if not hour_df.empty:
                    hour_df = add_traded_volume(hour_df)
                    # Keep only partial/matched trades
                    hour_df = hour_df[hour_df['ActionCode'].isin(['P', 'M'])]
                    hour_list.append(hour_df)

                # Process the quarter-hour trades
                if not qh_df.empty:
                    qh_df = add_traded_volume(qh_df)
                    # Keep only partial/matched trades
                    qh_df = qh_df[qh_df['ActionCode'].isin(['P', 'M'])]
                    quarter_hour_list.append(qh_df)
                
                pbar.update(1)
    
    # Concatenate all hour and quarter-hour data for the year
    combined_h_df = pd.concat(hour_list, ignore_index=True) if hour_list else pd.DataFrame(columns=necessary_columns)
    combined_qh_df = pd.concat(quarter_hour_list, ignore_index=True) if quarter_hour_list else pd.DataFrame(columns=necessary_columns)
    
    # Only keep columns: [side, deliverystart, transactiontime, price, volume traded]
    keep_cols = ['Side', 'DeliveryStart', 'TransactionTime', 'Price', 'VolumeTraded']
    
    # Hourly
    combined_h_df = combined_h_df[keep_cols]
    combined_h_df.to_csv(f"{year}_h_{country}.csv", index=False)
    
    # Quarter-hourly
    combined_qh_df = combined_qh_df[keep_cols]
    combined_qh_df.to_csv(f"{year}_qh_{country}.csv", index=False)


def merge_filtered_data(resolution, country):

    df_2022 = pd.read_csv('EPEX_Spot_Orderbook/'+f"2022_{resolution}_{country}.csv")
    df_2022.reset_index(drop=True, inplace=True)
    df_2022['DeliveryStart'] = pd.to_datetime(df_2022['DeliveryStart'])
    df_2022['TransactionTime'] = pd.to_datetime(df_2022['TransactionTime'])

    df_2023 = pd.read_csv('EPEX_Spot_Orderbook/'+f"2023_{resolution}_{country}.csv")
    df_2023.reset_index(drop=True, inplace=True)
    df_2023['DeliveryStart'] = pd.to_datetime(df_2023['DeliveryStart'])
    df_2023['TransactionTime'] = pd.to_datetime(df_2023['TransactionTime'])

    df_2024 = pd.read_csv('EPEX_Spot_Orderbook/'+f"2024_{resolution}_{country}.csv")
    df_2024.reset_index(drop=True, inplace=True)
    df_2024['DeliveryStart'] = pd.to_datetime(df_2024['DeliveryStart'])
    df_2024['TransactionTime'] = pd.to_datetime(df_2024['TransactionTime'])

    df = pd.concat([df_2022, df_2023, df_2024], ignore_index=True)
    df.to_pickle('EPEX_Spot_Orderbook/'+f"Filtered_{resolution}_{country}.pkl")


'''
-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
-- -- -- -- -- --   functions used for extracting features  -- -- -- -- -- -- --
-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
'''


def input_extraction(filtered_df):
    filtered_df = filtered_df.sort_values('TransactionTime')
    sum_volume = np.sum(filtered_df["VolumeTraded"])
    
    if sum_volume == 0:
        return (0, np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan, np.nan, np.nan)
    
    else:
        price_weighted_avg = np.average(filtered_df['Price'], weights=filtered_df['VolumeTraded'])
        min_price = np.min(filtered_df['Price'])
        max_price = np.max(filtered_df['Price'])
        last_price = filtered_df['Price'].iloc[-1]
        percentiles = np.percentile(filtered_df['Price'], [5, 25, 50, 75, 95])
        return (sum_volume, price_weighted_avg, min_price, max_price, last_price,
                *percentiles)


def extract_features(df, indice):
    data_per_file = []

    if indice == 'ID1':
        main_w = 60

    elif indice == 'ID2':
        main_w = 120

    elif indice == 'ID3':
        main_w = 180

    else:
        main_w = None
        print('Wrong indice, only ID1, ID2, or ID3')
    
    sub_windows = ['full', 180, 60, 15, 5, 1]
    total_groups = df['DeliveryStart'].nunique()

    with tqdm(total=total_groups, desc="Processing groups", unit="group") as pbar:
        for Date_DeliveryStart, group in df.groupby('DeliveryStart'):
            pbar.set_postfix_str(f"Processing date: {Date_DeliveryStart}")
            pbar.update(1)

            SumV_dic = {}
            VWAP_dic = {}
            MinP_dic = {}
            MaxP_dic = {}
            LastP_dic = {}
            P5_dic = {}
            P25_dic = {}
            P50_dic = {}
            P75_dic = {}
            P95_dic = {}

            end_dt = Date_DeliveryStart - pd.Timedelta(minutes=main_w)
            subwindow_values = {}

            for i, sub_w in enumerate(sub_windows):
                if sub_w == 'full':
                    df_sub = group[group['TransactionTime'] <= end_dt]
                else:
                    start_dt = end_dt - pd.Timedelta(minutes=sub_w)
                    df_sub = group[(group['TransactionTime'] >= start_dt) & (group['TransactionTime'] <= end_dt)]

                values = input_extraction(df_sub)

                if values[0] == 0:
                    found_nonzero = False
                    for bigger_sw in sub_windows[:i][::-1]:
                        prev_vals = subwindow_values.get(bigger_sw, None)
                        if prev_vals and prev_vals[0] != 0:
                            values = prev_vals
                            found_nonzero = True
                            break

                    if not found_nonzero:
                        values = (0, np.nan, np.nan, np.nan, np.nan,
                                    np.nan, np.nan, np.nan, np.nan, np.nan)

                subwindow_values[sub_w] = values

            for sub_w in sub_windows:
                (sumv_val, vwap_val, minp_val, maxp_val, lastp_val,
                    p5, p25, p50, p75, p95) = subwindow_values[sub_w]

                if sub_w == 'full':
                    prefix = 'full'
                else:
                    prefix = f'{sub_w}'

                SumV_dic[f'SumV_{prefix}'] = sumv_val
                VWAP_dic[f'VWAP_{prefix}'] = vwap_val
                MinP_dic[f'MinP_{prefix}'] = minp_val
                MaxP_dic[f'MaxP_{prefix}'] = maxp_val
                LastP_dic[f'LastP_{prefix}'] = lastp_val
                P5_dic[f'P5_{prefix}'] = p5
                P25_dic[f'P25_{prefix}'] = p25
                P50_dic[f'P50_{prefix}'] = p50
                P75_dic[f'P75_{prefix}'] = p75
                P95_dic[f'P95_{prefix}'] = p95

            row_dict = {
                'Date_DeliveryStart': Date_DeliveryStart,
                **SumV_dic,
                **VWAP_dic,
                **MinP_dic,
                **MaxP_dic,
                **LastP_dic,
                **P5_dic,
                **P25_dic,
                **P50_dic,
                **P75_dic,
                **P95_dic,
            }
            data_per_file.append(row_dict)

    result_df = pd.DataFrame(data_per_file)
    return result_df


def execute_feature_extraction(resolution, country, indice, side=True):

    # read data
    df = pd.read_pickle('EPEX_Spot_Orderbook/'+f"Filtered_{resolution}_{country}.pkl")
    df.reset_index(drop=True, inplace=True)
    
    # differentiate sides
    if side==True:
        # process buy side
        df_buy = extract_features(df[df["Side"] == "BUY"], indice)
        df_buy.to_pickle('EPEX_Spot_Orderbook/'+f"Feature_Buy_{resolution}_{country}_{indice}.pkl") 
        del df_buy

        # process sell side
        df_sell = extract_features(df[df["Side"] == "SELL"], indice)
        df_sell.to_pickle('EPEX_Spot_Orderbook/'+f"Feature_Sell_{resolution}_{country}_{indice}.pkl") 
        del df_sell

    # not differentiate sides
    elif side==False:
        df = extract_features(df, indice)
        df.to_pickle('EPEX_Spot_Orderbook/'+f"Feature_{resolution}_{country}_{indice}.pkl") 
        del df


'''
-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
-- -- -- -- -- -- functions used for extracting sequences   -- -- -- -- -- -- --
-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
'''


def extract_sequences(df, indice, max_points=256):
    all_data = []

    if indice == 'ID1':
        cutoff_minutes = 60

    elif indice == 'ID2':
        cutoff_minutes = 120

    elif indice == 'ID3':
        cutoff_minutes = 180

    else:
        cutoff_minutes = None
        print('Wrong indice, only ID1, ID2, or ID3')
    
    total_groups = df['DeliveryStart'].nunique()
    with tqdm(total=total_groups, desc="Extracting sequences", unit="group") as pbar:
        for Date_DeliveryStart, group in df.groupby('DeliveryStart'):
            pbar.set_postfix_str(f"Processing date: {Date_DeliveryStart}")
            pbar.update(1)

            end_dt = Date_DeliveryStart - pd.Timedelta(minutes=cutoff_minutes)
            filtered = group[group['TransactionTime'] <= end_dt].copy()

            if filtered.empty:
                continue

            filtered = filtered.sort_values('TransactionTime')

            # Extract sum of volume and number of matched trades
            sum_volume = np.sum(filtered['VolumeTraded'])
            num_trades = len(filtered)

            # Get only the latest N trades
            if len(filtered) > max_points:
                filtered = filtered.iloc[-max_points:]

            filtered['TimeDiffSec'] = (Date_DeliveryStart - filtered['TransactionTime']).dt.total_seconds()
            sequence = filtered[['Price', 'VolumeTraded', 'TimeDiffSec']].values.tolist()

            
            all_data.append({
                'Date_DeliveryStart': Date_DeliveryStart,
                'Sequence': sequence,
                'SumVolume': sum_volume,
                'NumTrades': num_trades
            })

    return pd.DataFrame(all_data)


def execute_sequence_extraction(resolution, country, indice, side=True):

    # Read data
    df = pd.read_pickle('EPEX_Spot_Orderbook/'+f"Filtered_{resolution}_{country}.pkl")
    df.reset_index(drop=True, inplace=True)
    
    # Differentiate sides
    if side==True:

        # Process buy side
        df_buy = extract_sequences(df[df["Side"] == "BUY"], indice)
        df_buy.to_pickle('EPEX_Spot_Orderbook/'+f"Sequence_Buy_{resolution}_{country}_{indice}.pkl") 
        del df_buy

        # Process sell side
        df_sell = extract_sequences(df[df["Side"] == "SELL"], indice)
        df_sell.to_pickle('EPEX_Spot_Orderbook/'+f"Sequence_Sell_{resolution}_{country}_{indice}.pkl") 
        del df_sell

    # Not differentiate sides
    elif side==False:
        df = extract_sequences(df, indice)
        df.to_pickle('EPEX_Spot_Orderbook/'+f"Sequence_NoSide_{resolution}_{country}_{indice}.pkl") 
        del df




'''
-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
-- -- -- -- -- --    functions used for extracting labels   -- -- -- -- -- -- --
-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
'''


def output_extraction(filtered_df):
    filtered_df = filtered_df.sort_values('TransactionTime')
    sum_volume = np.sum(filtered_df["VolumeTraded"])
    num_trades = len(filtered_df)

    if sum_volume == 0:
        return np.nan, 0, 0
    else:
        price_weighted_avg = np.average(filtered_df['Price'], weights=filtered_df['VolumeTraded'])
        return price_weighted_avg, sum_volume, num_trades


def extract_labels(df, country, indice):
    data_per_file = []

    if indice == 'ID1':
        start_offset = 60

    elif indice == 'ID2':
        start_offset = 120

    elif indice == 'ID3':
        start_offset = 180

    else:
        start_offset = None
        print('Wrong indice, only ID1, ID2, or ID3')

    if country == 'germany':
        end_offset = 30

    elif country == 'austria':
        end_offset = 0

    else:
        end_offset = None
        print('Wrong country, only austria or germany')

    total_groups = df['DeliveryStart'].nunique()

    with tqdm(total=total_groups, desc="Extracting labels", unit="group") as pbar:
        for delivery_start, group in df.groupby('DeliveryStart'):
            pbar.update(1)
            label_row = {'Date_DeliveryStart': delivery_start}

            start_dt = delivery_start - pd.Timedelta(minutes=start_offset)
            end_dt = delivery_start - pd.Timedelta(minutes=end_offset)
            df_sub = group[(group['TransactionTime'] >= start_dt) & (group['TransactionTime'] <= end_dt)]

            vwap, sumv, num_trades = output_extraction(df_sub)
            label_row[indice] = vwap
            label_row[f'SumV_{indice}'] = sumv
            label_row[f'NumTrades_{indice}'] = num_trades

            data_per_file.append(label_row)

    return pd.DataFrame(data_per_file)


def execute_label_extraction(resolution, country, indice, side=False):
    df = pd.read_pickle('EPEX_Spot_Orderbook/' + f"Filtered_{resolution}_{country}.pkl")
    df.reset_index(drop=True, inplace=True)

    if side is True:
        # BUY side
        df_buy = extract_labels(df[df["Side"] == "BUY"], country, indice)
        df_buy.to_pickle('EPEX_Spot_Orderbook/' + f"Label_Buy_{resolution}_{country}_{indice}.pkl")
        del df_buy

        # SELL side
        df_sell = extract_labels(df[df["Side"] == "SELL"], country, indice)
        df_sell.to_pickle('EPEX_Spot_Orderbook/' + f"Label_Sell_{resolution}_{country}_{indice}.pkl")
        del df_sell

    elif side is False:
        df_labels = extract_labels(df, country, indice)
        df_labels.to_pickle('EPEX_Spot_Orderbook/' + f"Label_{resolution}_{country}_{indice}.pkl")
        del df_labels




'''
-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
-- -- -- -- -- --  functions used for obtaining global price scaler  -- -- -- --
-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
'''


def fit_and_save_price_scaler(country, resolution, train_start_date, train_end_date):
    # Load and prepare data
    df = pd.read_pickle(os.path.join('EPEX_Spot_Orderbook/', f"Filtered_{resolution}_{country}.pkl"))
    df.reset_index(drop=True, inplace=True)

    # Filter training data
    df_train = df[(df['DeliveryStart'] >= train_start_date) & (df['DeliveryStart'] < train_end_date)]

    # Fit scaler on price values only
    scaler = RobustScaler()
    scaler.fit(df_train[['Price']].values)

    # Save the scaler
    scaler_path = os.path.join('EPEX_Spot_Orderbook/', f"robust_scaler_{country}_{resolution}.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")