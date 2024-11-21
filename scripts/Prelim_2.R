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


train_index <- setdiff(c(1:2140), test_index)  
tokenized_data <- tokenized_data[train_index, ]

prelim1_test_predictions_labs <- prelim1_test_predictions_labs %>% 
  select(-bclass, -bclass.pred) %>% 
  rename(bigram_pred = pred)

final_test_tokens <- inner_join(prelim1_test_predictions_labs, token_test, by = '.id')
#save(final_test_tokens, file = 'data/preliminary/prelim2/final_test_tokens.RData')

save(token_train, file = 'data/preliminary/prelim2/token_train.RData')

# token_test_unigrams %>% bind_cols(test_preds)

prelim1_training_predictions_labs <- prelim1_training_predictions_labs %>% 
  select(-bclass, -bclass.pred) %>% 
  rename(bigram_pred = pred)

final_test_tokens <- inner_join(prelim1_test_predictions_labs, token_test, by = '.id')
#save(final_test_tokens, file = 'data/preliminary/prelim2/final_test_tokens.RData')

# START HERE 

#load('data/preliminary/prelim2/bigrams.RData')
#load('data/preliminary/prelim2/test_index.RData')

# saving training dataset
#save(token_train, file = 'data/preliminary/prelim2/token_train.RData')

# combine bigrams with predictions
# test
#load('data/preliminary/prelim1_test_predictions_df.RData') # labels
#load('data/preliminary/prelim2/token_testn.RData') # tokens

prelim1_test_predictions_labs <- prelim1_test_predictions_labs %>% 
  select(-bclass, -bclass.pred) %>% 
  rename(bigram_pred = pred)

final_test_tokens <- inner_join(prelim1_test_predictions_labs, token_test, by = '.id')
#save(final_test_tokens, file = 'data/preliminary/prelim2/final_test_tokens.RData')

# training
load('data/preliminary/prelim1_training_predictions_df.RData') # labels
#load('data/preliminary/prelim2/token_train.RData') # tokens

prelim1_training_predictions_labs <- prelim1_training_predictions_labs %>% 
  select(-bclass, -bclass.pred) %>% 
  rename(bigram_pred = pred)

final_train_tokens <- inner_join(prelim1_training_predictions_labs, token_train, by = '.id')
#save(final_train_tokens, file = 'data/preliminary/prelim2/final_train_tokens.RData')

load('data/preliminary/prelim2/final_train_tokens.RData')
load('data/preliminary/prelim2/final_test_tokens.RData')

# Isolate predictions
train_preds <- final_train_tokens %>%
  select(-.id, -bclass, -bigram_pred)
train_preds <- as.data.frame(lapply(train_preds, as.numeric))
train_labels <- final_train_tokens %>%
  select(.id, bclass)

test_preds <- final_test_tokens %>%
  select(-.id, -bclass, -bigram_pred)
test_preds <- as.data.frame(lapply(test_preds, as.numeric))
test_labels <- final_test_tokens %>%
  select(.id, bclass)


# projection
proj_out <- projection_fn(.dtm = train_preds, .prop = 0.7)
train_dtm_projected <- proj_out$data
save(proj_out, file = 'data/preliminary/prelim2/final_proj_out_corrected.RData')

train <- train_labels %>%
  bind_cols(train_dtm_projected) %>% 
  inner_join(prelim1_training_predictions_labs, by = '.id')

# store predictors and response as matrix and vector
x_train <- train %>% select(-bclass, -.id) %>% as.matrix()
y_train <- train_labels %>% pull(bclass)

alpha_enet <- 0.3
fit_reg <- glmnet(x = x_train, 
                  y = y_train, 
                  family = 'binomial',
                  alpha = alpha_enet)

# choose a constraint strength by cross-validation
cvout <- cv.glmnet(x = x_train, 
                   y = y_train, 
                   family = 'binomial',
                   alpha = alpha_enet)

# store optimal strength
lambda_opt <- cvout$lambda.min

test_dtm_projected <- reproject_fn(.dtm = test_preds, proj_out) %>% 
  bind_cols(test_labels) %>% 
  select(-bclass) %>% 
  inner_join(prelim1_test_predictions_labs, by = '.id') %>% 
  select(-.id)

x_test <- as.matrix(test_dtm_projected)

# compute predicted probabilities
preds <- predict(fit_reg, 
                 s = lambda_opt, 
                 newx = x_test,
                 type = 'response')

pred_df <- test_labels %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(pred = as.numeric(preds)) %>%
  mutate(bclass.pred = factor(pred > 0.5, 
                              labels = levels(bclass)))

#save(pred_df, file='data/preliminary/prelim2/bigrams_results_final.RData')

panel <- metric_set(sensitivity, 
                    specificity, 
                    accuracy, 
                    roc_auc)

# compute test set accuracy
pred_df %>% panel(truth = bclass, 
                             estimate = bclass.pred, 
                             pred, 
                             event_level = 'second')

# without headers
headers_pred_df %>% panel(truth = bclass, 
                          estimate = bclass.pred, 
                          pred, 
                          event_level = 'second')

# without headers
no_headers_pred_df %>% panel(truth = bclass, 
                          estimate = bclass.pred, 
                          pred, 
                          event_level = 'second')


