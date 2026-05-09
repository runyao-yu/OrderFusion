import os
import joblib
import time
import numpy as np
from itertools import combinations

import matplotlib.pyplot as plt
import imageio.v3 as iio
from natsort import natsorted
from PIL import Image
from IPython.display import clear_output

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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



def compute_interval_metrics(y_true, y_pred_array, quantiles):
    """
    Compute mean width and coverage for each symmetric quantile interval.
    y_true:       array (n_samples,)
    y_pred_array: array (n_samples, n_quantiles)
    quantiles:    list of quantile levels (sorted ascending)
    Returns:
        dict of per-interval metrics + average metrics
    """
    results = {}
    widths, coverages = [], []

    # find symmetric quantile pairs (e.g. 0.1–0.9, 0.25–0.75)
    for low_q in quantiles:
        high_q = 1 - low_q
        if low_q < 0.5 and high_q in quantiles:
            i_low = quantiles.index(low_q)
            i_high = quantiles.index(high_q)

            y_low = y_pred_array[:, i_low]
            y_high = y_pred_array[:, i_high]

            width = np.mean(y_high - y_low)
            coverage = np.mean((y_true >= y_low) & (y_true <= y_high))

            results[f"Width_{low_q:.2f}-{high_q:.2f}"] = float(width)
            results[f"Coverage_{low_q:.2f}-{high_q:.2f}_%"] = float(coverage * 100)

            widths.append(width)
            coverages.append(coverage)

    results["AvgIntervalWidth"] = float(np.mean(widths)) if widths else np.nan
    results["AvgCoverageRate_%"] = float(np.mean(coverages) * 100) if coverages else np.nan

    return results


def compute_interval_metrics(y_true, y_pred_array, quantiles):
    """
    Compute mean width and coverage for each symmetric quantile interval,
    plus Average Interval Coverage Error (AICE) and (optional) % version.

    y_true:       (n_samples,)
    y_pred_array: (n_samples, n_quantiles)
    quantiles:    sorted list of quantile levels
    """
    results = {}
    widths, coverages = [], []
    cov_errors = []

    for low_q in quantiles:
        high_q = 1 - low_q
        if low_q < 0.5 and high_q in quantiles:
            i_low = quantiles.index(low_q)
            i_high = quantiles.index(high_q)

            y_low = y_pred_array[:, i_low]
            y_high = y_pred_array[:, i_high]

            width = np.mean(y_high - y_low)
            coverage = np.mean((y_true >= y_low) & (y_true <= y_high))  # in [0,1]
            nominal = high_q - low_q                                    # e.g. 0.9-0.1 = 0.8
            cov_err = abs(coverage - nominal)

            results[f"Width_{low_q:.2f}-{high_q:.2f}"] = float(width)
            results[f"Coverage_{low_q:.2f}-{high_q:.2f}_%"] = float(coverage * 100)
            results[f"CovErr_{low_q:.2f}-{high_q:.2f}"] = float(cov_err)

            widths.append(width)
            coverages.append(coverage)
            cov_errors.append(cov_err)

    results["AvgIntervalWidth"] = float(np.mean(widths)) if widths else np.nan
    results["AvgCoverageRate_%"] = float(np.mean(coverages) * 100) if coverages else np.nan

    # average absolute deviation from nominal interval coverage
    results["AvgIntervalCoverageError"] = float(np.mean(cov_errors)) if cov_errors else np.nan
    results["AvgIntervalCoverageError_%"] = float(np.mean(cov_errors) * 100) if cov_errors else np.nan

    return results




def evaluate_model(best_model, X_test, y_test, quantiles, save_path, country, resolution):

    # Load scaler
    scaler_path = os.path.join(save_path, f"Data/scaler_{country}_{resolution}.pkl")
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
    interval_metrics = compute_interval_metrics(y_test_original, y_pred_array, quantiles)

    # Prepare results
    results = {
        'quantile_losses': quantile_losses,            
        'avg_quantile_loss': round(avg_quant_loss, 2), 
        'quantile_crossing_rate': round(crossing_rate * 100, 2),
        'coverage_error': interval_metrics['AvgIntervalCoverageError_%'],
        'median_quantile_rmse': round(rmse_median, 2),
        'median_quantile_mae': round(mae_median, 2),
        'median_quantile_r2': round(r2_median, 2),
        'inference_time': inference_time,
        'y_test_original': y_test_original,
        'y_pred_list': y_pred_list}

    
    results.update(interval_metrics)

    print(
        f"AQL: {results['avg_quantile_loss']}, "
        f"AQCR: {results['quantile_crossing_rate']}, "
        f"AQCE: {results['AvgIntervalCoverageError_%']}, "
        f"AIW: {results['AvgIntervalWidth']}, "
        f"RMSE: {results['median_quantile_rmse']}, "
        f"MAE: {results['median_quantile_mae']}, "
        f"R2: {results['median_quantile_r2']}, "
        f"Inference time: {inference_time}s \n"
    )

    return results


def quantile_loss(q, name):
    def loss(y_true, y_pred):
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(q * e, (q - 1) * e))
    loss.__name__ = f'{name}_label'
    return loss


def load_best_model(quantiles, country, resolution, indice, save_path):
    
    checkpoint_path = os.path.join(save_path, f"Model/OrderFusion_{country}_{resolution}_{indice}.keras")
    quantiles = [int(q * 100) for q in quantiles]
    quantiles_dict = {f'q{q:02}': q / 100 for q in quantiles}
    custom_objects = {f'{name}_label': quantile_loss(q, name) for name, q in quantiles_dict.items()}

    with custom_object_scope(custom_objects):
        best_model = load_model(checkpoint_path, custom_objects=custom_objects)
    return best_model


def get_forecasts(best_model, save_path, X_test, y_test, quantiles, country, resolution):
    # Load scaler
    scaler_path = os.path.join(save_path, f"Data/scaler_{country}_{resolution}.pkl")
    scaler = joblib.load(scaler_path)
    
    # Sort quantiles and scale back the true prices
    quantiles = sorted(quantiles)
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

    y_pred_scaled_list = best_model.predict(X_test)  # list/array of shape [n_quantiles, n_samples]

    # Scale back each quantile prediction
    y_pred_list = []
    for i, q in enumerate(quantiles):
        pred_rescaled = scaler.inverse_transform(y_pred_scaled_list[i].reshape(-1, 1)).ravel()
        y_pred_list.append(pred_rescaled)
    return y_pred_list, y_test_original


def plot_forecasts(y_pred_list, y_test_original, window_size, stop_index, indice):
    
    y_q10, y_q50, y_q90 = y_pred_list
    N = len(y_test_original)
    assert all(len(arr) == N for arr in [y_q10, y_q50, y_q90]), "Input arrays must have the same length"

    
    for i in range(N - window_size + 1):

        if i + window_size > stop_index+1:
            break

        x = np.arange(i, i + window_size)

        y_true_win = y_test_original[i:i + window_size]
        y_q10_win = y_q10[i:i + window_size]
        y_q50_win = y_q50[i:i + window_size]
        y_q90_win = y_q90[i:i + window_size]

        # Set dynamic y-limits with padding and determine 4 y-ticks
        y_min, y_max = np.min([y_true_win, y_q10_win, y_q90_win]), np.max([y_true_win, y_q10_win, y_q90_win])
        y_pad_min = y_min * 1.1 if y_min < 0 else y_min * 0.9
        y_pad_max = y_max * 1.1 if y_max > 0 else y_max * 0.9
        
        # Plot
        clear_output(wait=True)
        fig, ax = plt.subplots(figsize=(3.6, 2.3))
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 13

        ax.plot(x, y_true_win, label="True", color="black", linewidth=1.1)
        ax.plot(x, y_q50_win, label=r"Q$_{0.5}$", color="#F05D06", linewidth=1.1, alpha=0.9)
        ax.fill_between(x, y_q10_win, y_q90_win, alpha=0.9, label=r"Q$_{0.1}$-Q$_{0.9}$", color="#6A6657", linewidth=1.1)

                
        ax.set_ylabel(f"ID$_{indice[-1]}$ (€/MWh)")
        yticks = np.linspace(y_pad_min, y_pad_max, 4)
        ax.set_yticks(yticks)
        ax.set_ylim(yticks[0], yticks[-1])

        ax.set_xlabel("Testing sample index")
        xticks = np.linspace(np.min(x), np.max(x), 6)
        ax.set_xticks(xticks)
        ax.set_xlim(np.min(x), np.max(x))

        ax.tick_params(axis='both', direction='out', width=1.1)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_linewidth(1.1)
        ax.spines['bottom'].set_linewidth(1.1)

        fig.legend(
            loc="lower center",
            ncol=3,
            fontsize=11,
            frameon=False,
            bbox_to_anchor=(0.6, -0.07)
        )
        plt.tight_layout()
        plt.savefig(f'Figure/{indice}/{indice}_{i}.png', dpi=300, bbox_inches='tight', transparent=True)
        plt.show()
        time.sleep(0.00001)  


def create_gif_from_images(image_dir, output_path, prefix="", duration=0.01, size=None):

    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                   if f.endswith(".png") and f.startswith(prefix)]
    image_files = natsorted(image_files)

    if not image_files:
        raise ValueError(f"No matching images in {image_dir} with prefix '{prefix}'")

    frames = []
    for img_path in image_files:
        img = Image.open(img_path).convert("RGB")
        if size:
            img = img.resize(size, Image.LANCZOS)  # resize to target size
        frames.append(np.asarray(img)) 

    iio.imwrite(output_path, frames, format="gif", duration=duration)


def gif_conversion(indice):
    first_image = Image.open(f"Figure/{indice}/{indice}_0.png")
    target_size = first_image.size 

    create_gif_from_images(
        image_dir=f"Figure/{indice}",
        output_path=f"Figure/{indice}_GIF.gif",
        prefix=f"{indice}_",
        duration=0.0005,
        size=target_size 
    )
