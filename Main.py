from OrderFusion import *

# 'local' if run locally or 'cloud' run on Google Colab
save_path = train_status('local') 

# markets: 'germany' or  'austria' 
countries = ['austria', 'germany'] 

# product types: 'h' stands for hourly
resolutions = ['h'] 

# price indice: 'ID3', 'ID2', 'ID1'
indices = ['ID3', 'ID2', 'ID1']  

# start date for training data; e.g. (24, 3, 3): 24 months for training, 3 months for validation, 3 months for testing
train_start_date = '2022-01-01'
split_lens = [(24, 6, 6)]
years = [2022, 2023, 2024]

# target quantiles to forecast
quantiles = [0.1, 0.5, 0.9] 

# optimized hidden_dimension, max. degree of interactions, number of attention heads
model_shapes = [[3, 4, 1]] 

# recommended number of epochs and batch size are 50 and 256 respectively
epoch, batch_size = 50, 256 

# optimized number of trades (L in the paper) in the input sequence to retain
points = [16] 
''' Optimized values are:
    German market: 16 for ID1, ID2, and ID3
    Austrian market: 8 for ID1, ID2, and 2 for ID3
'''

# random seeds
seeds = [42] 

# show_progress_bar or not during training and validation
show_progress_bar = False 

model_modes = ['OrderFusion']

# configuration
data_config = (countries, resolutions, indices, save_path, split_lens, train_start_date)
model_config = (model_modes, model_shapes, epoch, batch_size, points, quantiles, seeds, show_progress_bar)

# whether to process the orderbook data or not
processing = True
'''set to True if first time running the code, 
   intermediate processed data will be saved in the 'Data' folder,
   otherwise set to False
'''

# main function to run all steps
if __name__ == "__main__":

    if processing == True:
        split_len = (24, 6, 6) # replace it corresponding to your actual train/val/test split setup, e.g. (24, 6, 6) 
        processing_orderbook(countries, years, resolutions, indices, train_start_date, get_train_end_date(train_start_date, split_len))
        execute_main(data_config, model_config)

    else: 
        execute_main(data_config, model_config)
        



