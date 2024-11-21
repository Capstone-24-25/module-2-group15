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
# no_headers <- load('data/claims-clean.RData') # original data
headers_clean <- load('data/prelim1_data.RData') # data with header tags included

set.seed(112233)
# tokenize, split, and project

# tokenize the data
tokenized_data <- nlp_fn(prelim1_data)

# Set class as a factor
tokenized_data$bclass <- as.factor(tokenized_data$bclass)

# Split 
token_split <- initial_split(tokenized_data, .8)
token_train <- training(token_split)
token_test <- testing(token_split)

# Isolate predictions
train_preds <- token_train %>%
  select(-.id, -bclass)
train_preds <- as.data.frame(lapply(train_preds, as.numeric))
train_labels <- token_train %>%
  select(.id, bclass)

test_preds <- token_test %>%
  select(-.id, -bclass)
test_preds <- as.data.frame(lapply(test_preds, as.numeric))
test_labels <- token_test %>%
  select(.id, bclass)

save(token_test, file ="./data/preliminary/prelim2/token_test_unigrams.RData")
save(token_train, file ="./data/preliminary/prelim2/token_train_unigrams.RData")

# projection
proj_out <- projection_fn(.dtm = train_preds, .prop = 0.7)
train_dtm_projected <- proj_out$data

train <- train_labels %>%
  bind_cols(train_dtm_projected)

# store predictors and response as matrix and vector
x_train <- train %>% select(-bclass, -.id) %>% as.matrix()
y_train <- train_labels %>% pull(bclass)

# fit enet model
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

# view results
cvout
  
# project test data onto PCs
test_dtm_projected <- reproject_fn(.dtm = test_preds, proj_out)

# coerce to matrix
x_test <- as.matrix(test_dtm_projected)

# compute predicted probabilities
preds <- predict(fit_reg, 
                 s = lambda_opt, 
                 newx = x_test,
                 type = 'response')

prelim1_training_predictions <- predict(fit_reg, 
                 s = lambda_opt, 
                 newx = x_train,
                 type = 'response')

prelim1_test_predictions <- preds 

#save(prelim1_training_predictions, file ="./data/preliminary/prelim1_training_predictions.RData")
#save(prelim1_test_predictions, file ="./data/preliminary/prelim1_test_predictions.RData")

# store predictions in a data frame with true labels
pred_df <- test_labels %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(pred = as.numeric(preds)) %>%
  mutate(bclass.pred = factor(pred > 0.5, 
                              labels = levels(bclass)))

prelim1_test_predictions_labs <- test_labels %>%
  mutate(bclass = factor(bclass)) %>%
  bind_cols(pred = as.numeric(preds)) %>%
  mutate(bclass.pred = factor(pred > 0.5, 
                              labels = levels(bclass)))

prelim1_training_predictions_labs <- train_labels %>%
  mutate(bclass = factor(bclass)) %>%
  bind_cols(pred = as.numeric(prelim1_training_predictions)) %>%
  mutate(bclass.pred = factor(pred > 0.5, 
                              labels = levels(bclass)))
save(prelim1_training_predictions_labs, file ="./data/preliminary/prelim1_training_predictions_df.RData")
save(prelim1_test_predictions_labs, file ="./data/preliminary/prelim1_test_predictions_df.RData")

prelim1_test_predictions
# no_headers_fit <- fit_reg
# no_headers_lambda <- lambda_opt
# no_headers_pred_df <- pred_df 

headers_fit <- fit_reg
headers_lambda <- lambda_opt
headers_pred_df <- pred_df

#save(no_headers_pred_df, file ="./data/preliminary/no_headers_pred_df.RData")
save(headers_pred_df, file ="./data/preliminary/headers_pred_df.RData")

# define classification metric panel 
panel <- metric_set(sensitivity, 
                    specificity, 
                    accuracy, 
                    roc_auc)

# compute test set accuracy
no_headers_pred_df %>% panel(truth = bclass, 
                  estimate = bclass.pred, 
                  pred, 
                  event_level = 'second')

headers_pred_df %>% panel(truth = bclass, 
                             estimate = bclass.pred, 
                             pred, 
                             event_level = 'second')

