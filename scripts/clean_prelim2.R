require(tidyverse)
require(keras)
require(tensorflow)
require(rsample)
require(sparsesvd)
require(glmnet)
require(modelr)
require(Matrix)
require(tidymodels)

# import svd wrapper
source('https://raw.githubusercontent.com/pstat197/pstat197a/main/materials/activities/data/projection-functions.R')

# import preprocessing scripts
source('scripts/preprocessing.R')

# data sources
headers_clean <- load('data/prelim1_data.RData') # data with header tags included

# tokenized bigrams
load('data/preliminary/prelim2/bigrams.RData')

# import predictions
load('data/preliminary/prelim1_training_predictions_df.RData')
load('data/preliminary/prelim1_test_predictions_df.RData')

# combine unigram-based prediction for training and test sets
token_train <- tokenized_data %>% 
  inner_join(prelim1_training_predictions_labs, by = '.id') %>% 
  select(-bclass, -bclass.pred)

token_test <- tokenized_data %>% 
  inner_join(prelim1_test_predictions_labs, by = '.id') %>% 
  select(-bclass, -bclass.pred)

prelim1_test_predictions_df

# load prelim log-odds
prelim1_train_labels <- prelim1_train_labels %>% 
  select(-bclass)

tokenized_data %>% 
  anti_join(prelim1_train_labels, by = '.id')

log_odds <- prelim1_train_labels %>% 
  bind_cols(prelim1_training_predictions) %>% 
  rename(log_odds = s1)

# bind predictions to bigram data
token_train_2 <- tokenized_data %>% 
  right_join(log_odds, by='.id')

token_test_2 <- tokenized_data %>% 
  anti_join(prelim1_train_labels, by = '.id')

  token_test_2 %>% 
  bind_cols(pre
            
# token_test_unigrams %>% bind_cols(test_preds)
