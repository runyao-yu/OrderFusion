import os
import joblib
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler

def add_traded_volume(df):
    """
    Calculates the traded volume per transaction within each OrderId group.
    Only rows with non-zero, non-NaN VolumeTraded remain at the end.
    """
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


def filter_data(country, year):
    base_path = f"Data/{country}/Intraday Continuous/Orders"
    
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
    
    with tqdm(total=total_files, desc=f"  processing {year} data for {country}", unit="file") as pbar:
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
    keep_cols = ['Side', 'DeliveryStart', 'TransactionTime', 'Price', 'VolumeTraded']
    
    # Hourly
    combined_h_df = combined_h_df[keep_cols]
    combined_h_df.to_csv(f"Data/{year}_h_{country}.csv", index=False)
    
    # Quarter-hourly
    combined_qh_df = combined_qh_df[keep_cols]
    combined_qh_df.to_csv(f"Data/{year}_qh_{country}.csv", index=False)


def filter_data(country, start_date, end_date):  # CHANGED: signature -> dates not year
    base_path = f"Data/{country}/Intraday Continuous/Orders"
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

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    months = pd.date_range(start=start, end=end, freq='MS', inclusive='left')  # month starts

    # columns we keep in the outputs (moved up so we can use for empty dfs)
    keep_cols = ['Side', 'DeliveryStart', 'TransactionTime', 'Price', 'VolumeTraded']

    # pre-count files across all selected months (for a single tqdm)
    total_files = 0
    for m in months:
        mdir = os.path.join(base_path, str(m.year), f"{m.month:02d}")
        if os.path.isdir(mdir):
            total_files += sum(len(files) for _, _, files in os.walk(mdir))

    with tqdm(total=total_files,
              desc=f"  processing {country} {start_date} → {end_date}",
              unit="file") as pbar:

        # process month by month and save two files per month
        for m in months:
            year_str = str(m.year)
            month_str = f"{m.month:02d}"
            path = os.path.join(base_path, year_str, month_str)

            if not os.path.isdir(path):
                continue  

            # collect results per-month 
            hour_list = []
            quarter_hour_list = []

            for dirname, _, filenames in os.walk(path):
                for filename in filenames:
                    data_path = os.path.join(dirname, filename)

                    # read CSV with only necessary columns
                    df = pd.read_csv(
                        data_path,
                        header=1,
                        dtype={'ParentId': 'Int64'},
                        usecols=necessary_columns
                    )

                    # split into hour and quarter-hour subsets 
                    hour_df = df[df['Product'].isin(['Intraday_Hour_Power', 'XBID_Hour_Power'])]
                    qh_df   = df[df['Product'].isin(['Intraday_Quarter_Hour_Power', 'XBID_Quarter_Hour_Power'])]

                    # process the hour trades 
                    if not hour_df.empty:
                        hour_df = add_traded_volume(hour_df)
                        hour_df = hour_df[hour_df['ActionCode'].isin(['P', 'M'])]
                        hour_list.append(hour_df)

                    # process the quarter-hour trades
                    if not qh_df.empty:
                        qh_df = add_traded_volume(qh_df)
                        qh_df = qh_df[qh_df['ActionCode'].isin(['P', 'M'])]
                        quarter_hour_list.append(qh_df)

                    pbar.update(1)

            # concatenate per-month data 
            empty_df = pd.DataFrame(columns=keep_cols) 

            combined_h_df  = pd.concat(hour_list, ignore_index=True) if hour_list else empty_df.copy()
            combined_qh_df = pd.concat(quarter_hour_list, ignore_index=True) if quarter_hour_list else empty_df.copy()

            # keep only selected columns
            combined_h_df  = combined_h_df[keep_cols]
            combined_qh_df = combined_qh_df[keep_cols]

            # save data
            #combined_h_df.to_csv(f"Data/{year_str}_{month_str}_h_{country}.csv", index=False)
            #combined_qh_df.to_csv(f"Data/{year_str}_{month_str}_qh_{country}.csv", index=False)
            combined_h_df.to_pickle(f"Data/raw_{year_str}_{month_str}_h_{country}.pkl")
            combined_qh_df.to_pickle(f"Data/raw_{year_str}_{month_str}_qh_{country}.pkl")


def merge_data(resolution, country):
    dfs = []
    # look for .pkl files
    pattern = os.path.join('Data', f"raw_*_{resolution}_{country}.pkl")
    file_list = sorted(glob.glob(pattern))

    if not file_list:
        print(f"No files found: {pattern}")
        return
    print("Merging files:\n  - " + "\n  - ".join(os.path.basename(p) for p in file_list))

    for path in file_list:
        df_year = pd.read_pickle(path)
        df_year.reset_index(drop=True, inplace=True)
        df_year['DeliveryStart']   = pd.to_datetime(df_year['DeliveryStart'])
        df_year['TransactionTime'] = pd.to_datetime(df_year['TransactionTime'])
        dfs.append(df_year)

    df = pd.concat(dfs, ignore_index=True)
    df.to_pickle(os.path.join('Data', f"processed_{resolution}_{country}.pkl"))


# ------------------------------------------------------------------------------------------------
def extract_sequence(df, indice, max_points=256):
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
    with tqdm(total=total_groups, desc="  Extracting sequences", unit="group") as pbar:
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


def get_input(resolution, country, indice, side=True):

    # Read data
    df = pd.read_pickle('Data/'+f"processed_{resolution}_{country}.pkl")
    df.reset_index(drop=True, inplace=True)
    
    # Differentiate sides
    if side==True:

        # Process buy side
        df_buy = extract_sequence(df[df["Side"] == "BUY"], indice)
        df_buy.to_pickle('Data/'+f"sequence_buy_{resolution}_{country}_{indice}.pkl") 
        del df_buy

        # Process sell side
        df_sell = extract_sequence(df[df["Side"] == "SELL"], indice)
        df_sell.to_pickle('Data/'+f"sequence_sell_{resolution}_{country}_{indice}.pkl") 
        del df_sell

    # Not differentiate sides
    elif side==False:
        df = extract_sequence(df, indice)
        df.to_pickle('Data/'+f"sequence_{resolution}_{country}_{indice}.pkl") 
        del df


def output_extraction(filtered_df):
    filtered_df = filtered_df.sort_values('TransactionTime')
    sum_volume = np.sum(filtered_df["VolumeTraded"])
    num_trades = len(filtered_df)

    if sum_volume == 0:
        return np.nan, 0, 0
    else:
        price_weighted_avg = np.average(filtered_df['Price'], weights=filtered_df['VolumeTraded'])
        return price_weighted_avg, sum_volume, num_trades


# ------------------------------------------------------------------------------------------------
def extract_label(df, country, indice):
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

    with tqdm(total=total_groups, desc="  Extracting labels", unit="group") as pbar:
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


def get_output(resolution, country, indice, side=False):
    df = pd.read_pickle('Data/' + f"processed_{resolution}_{country}.pkl")
    df.reset_index(drop=True, inplace=True)

    if side is True:
        # BUY side
        df_buy = extract_label(df[df["Side"] == "BUY"], country, indice)
        df_buy.to_pickle('Data/' + f"label_buy_{resolution}_{country}_{indice}.pkl")
        del df_buy

        # SELL side
        df_sell = extract_label(df[df["Side"] == "SELL"], country, indice)
        df_sell.to_pickle('Data/' + f"label_sell_{resolution}_{country}_{indice}.pkl")
        del df_sell

    elif side is False:
        df_labels = extract_label(df, country, indice)
        df_labels.to_pickle('Data/' + f"label_{resolution}_{country}_{indice}.pkl")
        del df_labels


def get_scaler(country, resolution, train_start_date, train_end_date):
    # Load and prepare data
    df = pd.read_pickle(os.path.join('Data/', f"processed_{resolution}_{country}.pkl"))
    df.reset_index(drop=True, inplace=True)

    # Filter training data
    df_train = df[(df['DeliveryStart'] >= train_start_date) & (df['DeliveryStart'] < train_end_date)]

    # Fit scaler on price values only
    scaler = RobustScaler()
    scaler.fit(df_train[['Price']].values)

    # Save the scaler
    scaler_path = os.path.join('Data/', f"scaler_{country}_{resolution}.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler saved to {scaler_path}")


def device_choice(choice):

    if choice == "cloud":
        from google.colab import drive
        drive.mount('/content/drive')
        pre_path = "/content/drive/My Drive/OrderFusion/"

    elif choice == "local":
        pre_path = os.path.abspath(".") + "/"
        
    if not os.path.exists(pre_path):
        os.makedirs(pre_path)

    return pre_path


def read_data(save_path, country, resolution, indice):

    # Load labels
    output = pd.read_pickle(f"{save_path}Data/label_{resolution}_{country}_{indice}.pkl")
    output = output[['Date_DeliveryStart', f'{indice}']]

    # Load sequences
    input_buy = pd.read_pickle(f"{save_path}Data/sequence_buy_{resolution}_{country}_{indice}.pkl")
    input_sell = pd.read_pickle(f"{save_path}Data/sequence_sell_{resolution}_{country}_{indice}.pkl")

    input_buy = input_buy.rename(columns={"Sequence": "Sequence_Buy"})
    input_sell = input_sell.rename(columns={"Sequence": "Sequence_Sell"})

    # Merge features and labels
    input = pd.merge(input_buy, input_sell, on="Date_DeliveryStart", how="outer")
    merged = pd.merge(input, output, on="Date_DeliveryStart", how="outer")

    # Standardize time zone
    merged['UTC'] = pd.to_datetime(merged['Date_DeliveryStart'], utc=True)
    merged = merged.drop(columns=['Date_DeliveryStart'])
    merged.ffill(inplace=True)

    return merged


def split_data(orderbook_df, output_col, train_start, train_end, val_start, val_end, test_start, test_end):

    # Split into train, val, test
    train_start_date = (pd.to_datetime(train_start)).strftime('%Y-%m-%d')
    train_end_date =   (pd.to_datetime(train_end)).strftime('%Y-%m-%d')
    val_start_date =   (pd.to_datetime(val_start)).strftime('%Y-%m-%d')
    val_end_date =     (pd.to_datetime(val_end)).strftime('%Y-%m-%d')
    test_start_date =  (pd.to_datetime(test_start)).strftime('%Y-%m-%d')
    test_end_date =    (pd.to_datetime(test_end)).strftime('%Y-%m-%d')


    train_df = orderbook_df[(orderbook_df['UTC'] >= train_start_date) & (orderbook_df['UTC'] < train_end_date)]
    val_df = orderbook_df[(orderbook_df['UTC'] >= val_start_date) & (orderbook_df['UTC'] < val_end_date)]
    test_df = orderbook_df[(orderbook_df['UTC'] >= test_start_date) & (orderbook_df['UTC'] < test_end_date)]

    # Keep buy/sell separate
    X_train_buy = [np.array(seq) for seq in train_df['Sequence_Buy']]
    X_train_sell = [np.array(seq) for seq in train_df['Sequence_Sell']]

    X_val_buy = [np.array(seq) for seq in val_df['Sequence_Buy']]
    X_val_sell = [np.array(seq) for seq in val_df['Sequence_Sell']]

    X_test_buy = [np.array(seq) for seq in test_df['Sequence_Buy']]
    X_test_sell = [np.array(seq) for seq in test_df['Sequence_Sell']]

    y_train = train_df[output_col].values
    y_val = val_df[output_col].values
    y_test = test_df[output_col].values

    return (X_train_buy, X_train_sell), y_train, (X_val_buy, X_val_sell), y_val, (X_test_buy, X_test_sell), y_test




def pad_sequence(seq, def_len, pad_value):
    seq = np.array(seq)
    seq_len = len(seq)

    if seq_len >= def_len:
        return seq[-def_len:]  # take last def_len elements
    else:
        pad = np.full((def_len - seq_len, seq.shape[1]), pad_value)
        return np.vstack([pad, seq])  # pre-padding


def pad_dataset(X_buy, X_sell, def_len, pad_value=10000.0):
    X_buy_padded = [pad_sequence(seq, def_len, pad_value) for seq in X_buy]
    X_sell_padded = [pad_sequence(seq, def_len, pad_value) for seq in X_sell]
    return np.array(X_buy_padded), np.array(X_sell_padded)


def pack_dual_input_to_4d(buy_data, sell_data):
    """
    buy_data: np.array of shape (batch, seq_len, 3)
    sell_data: np.array of shape (batch, seq_len, 3)
    
    Returns:
        4D tensor with shape (batch, seq_len, 3, 2)
    """
    buy_data = np.expand_dims(buy_data, axis=-1)   # (batch, seq_len, 3, 1)
    sell_data = np.expand_dims(sell_data, axis=-1) # (batch, seq_len, 3, 1)

    return np.concatenate([buy_data, sell_data], axis=-1)  # (batch, seq_len, 3, 2)



def pad_data(X_train, X_val, X_test, num_trade, pad_value):

    X_train_buy, X_train_sell = X_train
    X_val_buy, X_val_sell = X_val
    X_test_buy, X_test_sell = X_test

    # Truncate and pad orderbook 
    X_train_buy_pad, X_train_sell_pad = pad_dataset(X_train_buy, X_train_sell, num_trade, pad_value)
    X_val_buy_pad, X_val_sell_pad = pad_dataset(X_val_buy, X_val_sell, num_trade, pad_value)
    X_test_buy_pad, X_test_sell_pad = pad_dataset(X_test_buy, X_test_sell, num_trade, pad_value)
    
    # Combine sides (buy and sell) 
    X_train = pack_dual_input_to_4d(X_train_buy_pad, X_train_sell_pad)
    X_val = pack_dual_input_to_4d(X_val_buy_pad, X_val_sell_pad)
    X_test = pack_dual_input_to_4d(X_test_buy_pad, X_test_sell_pad)

    return X_train, X_val, X_test


def scale_data(X_train, y_train, X_val, y_val, X_test, y_test, save_path, country, resolution):

    # Unpack buy and sell
    X_train_buy, X_train_sell = X_train
    X_val_buy, X_val_sell = X_val
    X_test_buy, X_test_sell = X_test

    # Stack all sequences together for global fitting
    flat_train = np.vstack(X_train_buy + X_train_sell)
    
    # Fit shared scaler over all 3 features: price, volume, Δt
    x_scaler = RobustScaler()
    x_scaler.fit(flat_train)


    def transform_sequences(X, scaler):
        return [scaler.transform(seq) for seq in X]

    # Perform scaling
    X_train_buy_scaled = transform_sequences(X_train_buy, x_scaler)
    X_train_sell_scaled = transform_sequences(X_train_sell, x_scaler)

    X_val_buy_scaled = transform_sequences(X_val_buy, x_scaler)
    X_val_sell_scaled = transform_sequences(X_val_sell, x_scaler)

    X_test_buy_scaled = transform_sequences(X_test_buy, x_scaler)
    X_test_sell_scaled = transform_sequences(X_test_sell, x_scaler)

    # Load the fitted RobustScaler
    scaler_path = os.path.join(save_path, f"Data/scaler_{country}_{resolution}.pkl")
    scaler = joblib.load(scaler_path)

    y_train_scaled = scaler.transform(np.array(y_train).reshape(-1, 1)).ravel()
    y_val_scaled = scaler.transform(np.array(y_val).reshape(-1, 1)).ravel()
    y_test_scaled = scaler.transform(np.array(y_test).reshape(-1, 1)).ravel()

    X_train_scaled = (X_train_buy_scaled, X_train_sell_scaled)
    X_val_scaled = (X_val_buy_scaled, X_val_sell_scaled)
    X_test_scaled = (X_test_buy_scaled, X_test_sell_scaled)

    return X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled
