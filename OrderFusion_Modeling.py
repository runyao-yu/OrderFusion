import os
import joblib
import time
import numpy as np
import pandas as pd
import random
import warnings
from itertools import combinations

import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention, Conv1D, Dense, Input, Flatten, Add, Subtract, Lambda, Concatenate, Layer
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import custom_object_scope

import sklearn
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

print("""Authors' versions are:
Python version: 3.9.6  
NumPy version: 1.24.0
Pandas version: 2.2.3
Joblib version: 1.4.2
TensorFlow version: 2.16.2
Scikit-learn version: 1.5.2
""")

'''
print(f"""Your versions are:
Python version: {os.sys.version}
NumPy version: {np.__version__}
Pandas version: {pd.__version__}
Joblib version: {joblib.__version__}
TensorFlow version: {tf.__version__}
Scikit-learn version: {sklearn.__version__}
""")
'''

'''
-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
-- -- -- -- -- -- functions used for orderbook data loading -- -- -- -- -- -- -- 
-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
'''


def train_status(status):

    if status == "cloud":
        from google.colab import drive
        drive.mount('/content/drive')
        pre_path = "/content/drive/My Drive/OrderFusion/"

    elif status == "local":
        pre_path = os.path.abspath(".") + "/"
        
    if not os.path.exists(pre_path):
        os.makedirs(pre_path)

    return pre_path


def read_data(save_path, country, resolution, indice):

    # Load labels
    output = pd.read_pickle(f"{save_path}EPEX_Spot_Orderbook/Label_{resolution}_{country}_{indice}.pkl")
    output = output[['Date_DeliveryStart', f'{indice}']]

    # Load sequences
    input_buy = pd.read_pickle(f"{save_path}EPEX_Spot_Orderbook/Sequence_Buy_{resolution}_{country}_{indice}.pkl")
    input_sell = pd.read_pickle(f"{save_path}EPEX_Spot_Orderbook/Sequence_Sell_{resolution}_{country}_{indice}.pkl")

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


'''
-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
-- -- -- -- -- -- functions used for data splitting and scaling -- -- -- -- -- -- 
-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
'''


def orderbook_split(orderbook_df, train_start_date, split_len, output_col):

    # Split into train, val, test
    train_len, val_len, test_len = split_len

    train_start_date_dt = pd.to_datetime(train_start_date)
    train_end_date_dt = train_start_date_dt + pd.DateOffset(months=train_len)
    val_end_date_dt = train_end_date_dt + pd.DateOffset(months=val_len)
    test_end_date_dt = val_end_date_dt + pd.DateOffset(months=test_len)

    train_start_date = train_start_date_dt.strftime('%Y-%m-%d')
    train_end_date = train_end_date_dt.strftime('%Y-%m-%d')
    val_end_date = val_end_date_dt.strftime('%Y-%m-%d')
    test_end_date = test_end_date_dt.strftime('%Y-%m-%d')

    train_df = orderbook_df[(orderbook_df['UTC'] >= train_start_date) & (orderbook_df['UTC'] < train_end_date)]
    val_df = orderbook_df[(orderbook_df['UTC'] >= train_end_date) & (orderbook_df['UTC'] < val_end_date)]
    test_df = orderbook_df[(orderbook_df['UTC'] >= val_end_date) & (orderbook_df['UTC'] < test_end_date)]

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



def orderbook_scale(X_train, y_train, X_val, y_val, X_test, y_test, save_path, country, resolution):

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
    scaler_path = os.path.join(save_path, f"EPEX_Spot_Orderbook/robust_scaler_{country}_{resolution}.pkl")
    scaler = joblib.load(scaler_path)

    y_train_scaled = scaler.fit_transform(np.array(y_train).reshape(-1, 1)).ravel()
    y_val_scaled = scaler.transform(np.array(y_val).reshape(-1, 1)).ravel()
    y_test_scaled = scaler.transform(np.array(y_test).reshape(-1, 1)).ravel()

    X_train_scaled = (X_train_buy_scaled, X_train_sell_scaled)
    X_val_scaled = (X_val_buy_scaled, X_val_sell_scaled)
    X_test_scaled = (X_test_buy_scaled, X_test_sell_scaled)

    return X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled


'''
-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
-- -- -- -- -- --  functions used for data truncation and padding -- -- -- -- --
-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
'''

def pad_sequence(seq, def_len, pad_value=10000.0):
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


'''
-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
-- -- -- -- -- -- codes used for proposed OrderFusion -- -- -- -- -- -- --
-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
'''

def set_random_seed(seed_value):

    # Set random seed for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    tf.random.set_seed(seed_value)


def quantile_loss(q, name):
    def loss(y_true, y_pred):
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(q * e, (q - 1) * e))
    loss.__name__ = f'{name}_label'
    return loss


def lr_schedule(epoch):

    # Decay learning rate every 10 epochs
    initial_lr = 5e-4
    decay_factor = 0.9
    decay_interval = 10
    
    num_decays = epoch // decay_interval
    return initial_lr * (decay_factor ** num_decays)


def HierarchicalQuantileHeadQ50(shared_representation, quantiles):

    # Sort quantiles and find the index of the median
    sorted_quantiles = sorted(quantiles)
    median_index = sorted_quantiles.index(50)

    # Start with the median quantile
    output_median = Dense(1, name='q50_label')(shared_representation)
    outputs = {f'q50_label': output_median}
    
    # Process quantiles above the median
    prev_output = output_median
    for q in sorted_quantiles[median_index + 1:]:
        residual = Dense(1)(shared_representation)
        residual = Lambda(tf.nn.relu)(residual)
        output = Add(name=f'q{q:02}_label')([prev_output, residual])
        outputs[f'q{q:02}_label'] = output
        prev_output = output  

    # Process quantiles below the median in reverse order
    prev_output = output_median
    for q in reversed(sorted_quantiles[:median_index]):
        residual = Dense(1)(shared_representation)
        residual = Lambda(tf.nn.relu)(residual)
        output = Subtract(name=f'q{q:02}_label')([prev_output, residual])
        outputs[f'q{q:02}_label'] = output
        prev_output = output 
    
    return [outputs[f'q{q:02}_label'] for q in quantiles]


class MaskPaddedValues(Layer):
    def __init__(self, pad_value=10000.0, **kwargs):
        super().__init__(**kwargs)
        self.pad_value = pad_value

    def call(self, x):
        # x: (batch, time, feature)
        # mask: True where any feature ≠ pad_value
        mask = tf.reduce_any(tf.not_equal(x, self.pad_value), axis=-1, keepdims=True)  # (B, T, 1)
        return x * tf.cast(mask, x.dtype)  # zero out padded time steps
    

from keras.saving import register_keras_serializable

@register_keras_serializable()
class TimeStepMask(Layer):
    def __init__(self, pad_value=10000.0, **kwargs):
        super().__init__(**kwargs)
        self.pad_value = pad_value

    def call(self, x):
        # x: (batch, T, F)
        mask = tf.reduce_any(tf.not_equal(x, self.pad_value), axis=-1, keepdims=True)
        return tf.cast(mask, tf.float32)  # (batch, T, 1)
    

@register_keras_serializable()
class TemporalDecayMask(Layer):
    def __init__(self, decay_strength=1, **kwargs):
        """
        decay_strength: int, controls the number of recent time steps to keep.
                        Final mask keeps the last 2^decay_strength steps as 1, rest 0.
        """
        super().__init__(**kwargs)
        self.decay_strength = decay_strength

    def call(self, x):
        """
        x: Tensor of shape (B, T, F) — only shape[1] (T) is used
        returns: Binary mask of shape (B, T, 1)
        """
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]

        # Number of timesteps to keep
        cutoff_len = tf.minimum(T, 2 ** self.decay_strength)

        # Construct [0, ..., 0, 1, ..., 1] where last `cutoff_len` entries are 1
        mask_tail = tf.ones([cutoff_len], dtype=tf.float32)
        mask_head = tf.zeros([T - cutoff_len], dtype=tf.float32)
        mask_1d = tf.concat([mask_head, mask_tail], axis=0)  # (T,)
        mask_3d = tf.reshape(mask_1d, [1, T, 1])              # (1, T, 1)

        # Broadcast across batch
        return tf.tile(mask_3d, [B, 1, 1])  # (B, T, 1)


def cross_attn_jump_fusion(input_buy, input_sell, mask_buy, mask_sell, hidden_dim):
    # 1st 1D Conv (equivalent to Dense per time step) + mask
    db1 = Conv1D(hidden_dim, kernel_size=1, activation='swish')(input_buy)
    db1 = db1 * mask_buy
    ds1 = Conv1D(hidden_dim, kernel_size=1, activation='swish')(input_sell)
    ds1 = ds1 * mask_sell

    # 1st cross-attention + mask
    ca_b1 = MultiHeadAttention(
        num_heads=4, key_dim=hidden_dim // 4
    )(
        query=db1,
        key=ds1, value=ds1,
    )
    ca_b1 = ca_b1 * mask_buy

    ca_s1 = MultiHeadAttention(
        num_heads=4, key_dim=hidden_dim // 4
    )(
        query=ds1,
        key=db1, value=db1,
    )
    ca_s1 = ca_s1 * mask_sell

    # 2nd 1D Conv + mask
    db2 = Conv1D(hidden_dim, kernel_size=1, activation='swish')(ca_b1)
    db2 = db2 * mask_buy
    ds2 = Conv1D(hidden_dim, kernel_size=1, activation='swish')(ca_s1)
    ds2 = ds2 * mask_sell

    # 2nd cross-attention + mask
    ca_b2 = MultiHeadAttention(
        num_heads=4, key_dim=hidden_dim // 4
    )(
        query=db1,
        key=ds2, value=ds2,
    )
    ca_b2 = ca_b2 * mask_buy

    ca_s2 = MultiHeadAttention(
        num_heads=4, key_dim=hidden_dim // 4
    )(
        query=db2,
        key=db1, value=db1,
    )
    ca_s2 = ca_s2 * mask_sell

    return ca_b2, ca_s2


def OrderFusion(hidden_dim, num_block, input_shape, quantiles, decay_strength, pad_value=10000.0):
    model_input = Input(shape=input_shape, name='input')
    raw_buy  = model_input[..., 0]  # (B, T, F)
    raw_sell = model_input[..., 1]

    decay_mask = TemporalDecayMask(decay_strength)(raw_buy)  # or raw_sell, just to get the shape (B, T, 1)

    binary_mask_buy  = TimeStepMask(pad_value)(raw_buy)      # shape (B, T, 1)
    binary_mask_sell = TimeStepMask(pad_value)(raw_sell)     # shape (B, T, 1)

    # element-wise mask multiplication
    mask_buy  = binary_mask_buy  * decay_mask                # shape (B, T, 1)
    mask_sell = binary_mask_sell * decay_mask                # shape (B, T, 1)

    # broadcasted elementwise multiplication
    out_buy  = raw_buy  * mask_buy
    out_sell = raw_sell * mask_sell

    for _ in range(num_block):
        out_buy, out_sell = cross_attn_jump_fusion(
            out_buy, out_sell,
            mask_buy, mask_sell,
            hidden_dim
        )

    rep = Concatenate()([
        Flatten()(out_buy),
        Flatten()(out_sell)
    ])
    outputs = HierarchicalQuantileHeadQ50(rep, quantiles)
    return Model(inputs=model_input, outputs=outputs)


'''
-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
-- -- -- -- -- -- codes used for running experiments  -- -- -- -- -- -- -- 
-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
'''


def select_model(target_model, hidden_dim, num_block, input_shape, quantiles, decay_strength):

    if target_model == 'OrderFusion':
            model = OrderFusion(hidden_dim, num_block, input_shape, quantiles, decay_strength)
    
    else:
        raise ValueError(f"Unknown target_model: {target_model}")

    return model


def optimize_models(X_train, y_train, X_val, y_val, exp_setup):

    hidden_dim, num_blocks, epoch, batch_size, save_path, target_model, quantiles, decay_strength = exp_setup
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = select_model(target_model, hidden_dim, num_blocks, input_shape, quantiles, decay_strength)
    
    # Generate y_train_dict and y_val_dict
    y_train_dict = {f'q{q:02}_label': y_train for q in quantiles}
    y_val_dict = {f'q{q:02}_label': y_val for q in quantiles}
    quantiles_dict = {f'q{q:02}': q / 100 for q in quantiles}

    # Define quantile loss
    quantile_losses = {}
    for name, q in quantiles_dict.items():
        loss_name = f'{name}_label'
        quantile_losses[loss_name] = quantile_loss(q, name)

    model.compile(optimizer=Adam(learning_rate=1e-3), loss=quantile_losses)
    lr_scheduler = LearningRateScheduler(lr_schedule)

    # Count model params
    model_paras_count = model.count_params()
    print(f"paras: {model_paras_count}")

    # Validate model
    checkpoint_path = os.path.join(f"{save_path}Model", f"{target_model}.keras")
    checkpoint_callback = ModelCheckpoint(checkpoint_path,
                                          monitor='val_loss',
                                          save_freq="epoch",
                                          save_best_only=True,
                                          mode='min',
                                          verbose=0)
    
    history = model.fit(X_train, y_train_dict, 
                        epochs=epoch, verbose=0,
                        validation_data=(X_val, y_val_dict),
                        callbacks=[checkpoint_callback, lr_scheduler],
                        batch_size=batch_size)

    # Load the best model with lowest val loss
    custom_objects = {f'{name}_label': quantile_loss(q, name) for name, q in quantiles_dict.items()}
    with custom_object_scope(custom_objects):
        best_model = load_model(checkpoint_path, custom_objects=custom_objects)

    return best_model, history.history, model_paras_count


def pinball_loss(y_true, y_pred, quantile):
    errors = y_true - y_pred
    loss = np.where(errors >= 0, quantile * errors, (quantile - 1) * errors)
    return np.mean(loss)


def compute_quantile_losses(y_true, y_pred_list, quantiles):

    quantile_losses = []
    for q, y_pred in zip(quantiles, y_pred_list):
        loss = pinball_loss(y_true, y_pred, q)
        quantile_losses.append(loss)

    avg_quantile_loss = float(np.mean(quantile_losses))
    return quantile_losses, avg_quantile_loss


def compute_regression_metrics(y_true, y_pred):

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def compute_quantile_crossing_rate(y_pred_array):

    n_samples, n_quantiles = y_pred_array.shape
    index_pairs = list(combinations(range(n_quantiles), 2))

    # Count how many pairs are violations for each sample
    violation_counts = np.zeros(n_samples, dtype=int)

    for i, j in index_pairs:
        violations = y_pred_array[:, i] > y_pred_array[:, j]
        violation_counts += violations.astype(int)

    # If a sample has at least one violation, mark it as 1
    sample_has_crossing = (violation_counts > 0).astype(int)
    crossing_rate = sample_has_crossing.mean()
    return crossing_rate


def test_performance(best_model, X_test, y_test, quantiles, save_path, country, resolution):

    # Load scaler
    scaler_path = os.path.join(save_path, f"EPEX_Spot_Orderbook/robust_scaler_{country}_{resolution}.pkl")
    scaler = joblib.load(scaler_path)
    
    # Sort quantiles and scale back the true prices
    quantiles = sorted(quantiles)
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

    # Measure inference time
    start_time = time.time()
    y_pred_scaled_list = best_model.predict(X_test)  # list/array of shape [n_quantiles, n_samples]
    end_time = time.time()
    inference_time = end_time - start_time

    # Scale back each quantile prediction
    y_pred_list = []
    for i, q in enumerate(quantiles):
        pred_rescaled = scaler.inverse_transform(y_pred_scaled_list[i].reshape(-1, 1)).ravel()
        y_pred_list.append(pred_rescaled)

    # Compute metrics
    quantile_losses, avg_quant_loss = compute_quantile_losses(y_test_original, y_pred_list, quantiles)
    median_index = quantiles.index(0.5)
    median_predictions = y_pred_list[median_index]
    rmse_median, mae_median, r2_median = compute_regression_metrics(y_test_original, median_predictions)
    y_pred_array = np.column_stack(y_pred_list)
    crossing_rate = compute_quantile_crossing_rate(y_pred_array)

    # Prepare results
    results = {
        'quantile_losses': quantile_losses,            
        'avg_quantile_loss': round(avg_quant_loss, 2), 
        'quantile_crossing_rate': round(crossing_rate * 100, 2),
        'median_quantile_rmse': round(rmse_median, 2),
        'median_quantile_mae': round(mae_median, 2),
        'median_quantile_r2': round(r2_median, 2),
        'inference_time': inference_time,
        'y_test_original': y_test_original,
        'y_pred_list': y_pred_list}

    print(f"AQL: {results['avg_quantile_loss']}, AQCR: {results['quantile_crossing_rate']}, RMSE: {results['median_quantile_rmse']}, MAE: {results['median_quantile_mae']}, R2: {results['median_quantile_r2']}, Inference time: {inference_time}s \n")
    
    return results


def execute_main(data_config, model_config):
    countries, resolutions, indices, save_path, split_len, train_start_date = data_config
    target_model, model_shapes, epoch, batch_size, points, quantiles, seeds, decay_strength = model_config

    for country in countries: 
        for resolution in resolutions:
            for indice in indices:
                results_df = []

                # Read, split, and scale orderbook
                orderbook_df = read_data(save_path, country, resolution, indice)
                X_train, y_train, X_val, y_val, X_test, y_test = orderbook_split(orderbook_df, train_start_date, split_len, indice)
                X_train, y_train, X_val, y_val, X_test, y_test = orderbook_scale(X_train, y_train, X_val, y_val, X_test, y_test, save_path, country, resolution)
                X_train_buy, X_train_sell = X_train
                X_val_buy, X_val_sell = X_val
                X_test_buy, X_test_sell = X_test

                for point in points:
                    
                    # Truncate and pad orderbook 
                    X_train_buy_pad, X_train_sell_pad = pad_dataset(X_train_buy, X_train_sell, point)
                    X_val_buy_pad, X_val_sell_pad = pad_dataset(X_val_buy, X_val_sell, point)
                    X_test_buy_pad, X_test_sell_pad = pad_dataset(X_test_buy, X_test_sell, point)
                    
                    # Combine sides (bids and offers) 
                    X_train_pack = pack_dual_input_to_4d(X_train_buy_pad, X_train_sell_pad)
                    X_val_pack  = pack_dual_input_to_4d(X_val_buy_pad, X_val_sell_pad)
                    X_test_pack  = pack_dual_input_to_4d(X_test_buy_pad, X_test_sell_pad)

                    for model_shape in model_shapes:
                        
                        # Get model depth and width
                        hidden_dim, num_block  = model_shape[0], model_shape[1]

                        for seed in seeds:
                            set_random_seed(seed)
                            print(f'{country, resolution, indice} | point={point} | {target_model} | model_shape: {model_shape} | seed: {seed}')
                            
                            # Train, validate, and test model
                            exp_setup = (hidden_dim, num_block, epoch, batch_size, save_path, target_model, [int(q * 100) for q in quantiles], decay_strength)
                            best_model, hist_val, num_para = optimize_models(X_train_pack, y_train, X_val_pack, y_val, exp_setup)
                            min_val_loss = min(hist_val["val_loss"])
                            results = test_performance(best_model, X_test_pack, y_test, quantiles, save_path, country, resolution)
                            results_df.append({
                                'country': country, 'resolution': resolution, 'indice': indice, 'point': point, 'model_shape': model_shape, 'target_model': target_model, 'num_para': num_para, 'min_val_loss': min_val_loss, 'history': hist_val, 'seed': seed, 
                                'avg_q_loss': results['avg_quantile_loss'], 'quantile_crossing': results['quantile_crossing_rate'], 'rmse': results['median_quantile_rmse'], 'mae': results['median_quantile_mae'], 'r2': results['median_quantile_r2'], 'inference_time': results['inference_time'],
                                'y_test_original': results['y_test_original'], 'y_pred_list': results['y_pred_list']})

                results_df = pd.DataFrame(results_df)
                results_df.to_pickle(f"{save_path}Result/{country}_{resolution}_{indice}_{target_model}.pkl")
                results_df.to_csv(f"{save_path}Result/{country}_{resolution}_{indice}_{target_model}.csv")
