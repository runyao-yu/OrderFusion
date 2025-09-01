from OrderFusion import *

save_path = train_status('local') # 'local' if run locally or 'cloud' run on Google Colab
country = 'germany' # options: 'germany', 'austria' 
resolution = 'h' # options:  'h', 'qh'
indice = 'ID3' # options:  'ID3', 'ID2', 'ID1'
train_start_date = '2022-01-01' # start date for training data
split_len = (24, 6, 6) # e.g. (24, 6, 6): first 24 months for training, next 6 months for validation, next 6 months for testing
years = [2022, 2023, 2024] # years that you purschased the data for
quantiles = [0.1, 0.5, 0.9] # target quantiles to forecast, could be many e.g. [0.05, 0.1, 0.5, 0.9, 0.95]
model_shape = [16, 4, 1] # optimized hidden_dimension, max. degree of interactions, number of attention heads. Change might lead to unstable training.
epoch, batch_size = 50, 256 # optimized number of epochs and batch size. Change might lead to unstable training.
num_trade = 16 # optimized number of trades (L in the paper) in the input sequence to retain. Change might lead to unstable training.
''' Depends on markets:
    German market: 16 for ID1, ID2, and ID3
    Austrian market: 8 for ID1, ID2, and 2 for ID3
'''
seed = 42 # random seeds for reproducibility
show_progress_bar = True # whether to show the training progress
model_mode = 'OrderFusion'
data_config = (country, resolution, indice, save_path, split_len, train_start_date, years)
model_config = (model_mode, model_shape, epoch, batch_size, num_trade, quantiles, seed, show_progress_bar)
phase = 'all' # options: 'prepare', 'train', 'inference', 'all'
''' phase sequence: 'prepare' -> 'train' -> 'inference'
    if phase = 'all', it will run all three steps in order.
'''

# main function to run all steps
if __name__ == "__main__":
    execute_main(data_config, model_config, phase)