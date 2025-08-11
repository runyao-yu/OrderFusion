from OrderFusion import *

# 'local' if run locally or 'cloud' run on Google Colab
save_path = train_status('local') 

# markets: 'germany' or  'austria' 
countries = ['germany'] 

# product types: 'h' for hourly, 'qh' for quarter-hourly
resolutions = ['h'] 

# price indice: 'ID3', 'ID2', or 'ID1'
indices = ['ID2'] 

# start date for training data; 24 months for training, 6 months for validation, 6 months for testing
train_start_date, train_end_date, split_len = '2022-01-01', '2024-01-01', (24, 6, 6)  
years = [2022, 2023, 2024]

# target quantiles to forecast
quantiles = [0.1, 0.5, 0.9] 

# hidden_dimension and number of jump fusion blocks
model_shapes = [[64, 1], [64, 2], [64, 3]] 
model_shapes = [[64, 0]] 

# recommended number of epochs and batch size are 50 and 1024 respectively
epoch, batch_size = 50, 256 

# T_max or number of trades in the input sequence
points = [16] 
''' recommended T_max values are:
    German market: 64 for ID1, 16 for ID2, and 4 for ID3
    Austrian market: 16 for ID1, 4 for ID2, and 1 for ID3
'''

# random seeds
seeds = [42] 

# show_progress_bar or not during training and validation
show_progress_bar = True 

# configuration
data_config = (countries, resolutions, indices, save_path, split_len, train_start_date)
model_config = ('OrderFusion', model_shapes, epoch, batch_size, points, quantiles, seeds, show_progress_bar)

# whether to process the orderbook data or not
processing = False
'''set to True if first time running the code, 
   intermediate processed data will be saved in the 'Data' folder,
   otherwise set to False
'''

# main function to run all steps
if __name__ == "__main__":

    if processing == True:
        processing_orderbook(countries, years, resolutions, indices, train_start_date, train_end_date)
        execute_main(data_config, model_config)

    else: 
        execute_main(data_config, model_config)
        



