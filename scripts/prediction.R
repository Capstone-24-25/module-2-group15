require(keras3)
require(tensorflow)
require(dplyr)
require(tidyr)
require(tidymodels)
require(knitr)
require(tidyverse)
require(ranger)
require(tidytext)
require(textstem)
require(rvest)
require(qdapRegex)
require(vip)
require(stopwords)
require(tokenizers)
load("data/claims-test.RData")
load("data/claims-raw.RData")
load("data/prelim1_data.RData")
source("scripts/preprocessing.R")


########################################
#             Binary Prep              #
########################################

load("results/rf-binary.rda")
parsed_claims <- parse_data(claims_raw)
token_data <- nlp_fn(parsed_claims)
token_data$bclass <- as.factor(token_data$bclass)

split <- initial_split(token_data, prop = 0.75, strata = bclass)
training <- training(split)
testing <- testing(split)
folds <- vfold_cv(data = token_data, v = 3, strata = bclass)

recipe <- recipe(bclass ~ ., data = training %>% select(-.id)) %>%
  step_pca(all_predictors(), num_comp = 100) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_nzv(all_predictors())

rf_model <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = tune()
) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

rf_wrkflw <- workflow() %>%
  add_model(rf_model) %>%
  add_recipe(recipe)

rf_grid <- grid_random(
  mtry(range = c(1, 10)),
  trees(range = c(200, 400)),
  min_n(range = c(5, 10)),
  size = 20
)

rf_best <- select_best(rf_tune, metric = "roc_auc")
rf_final_binary <- finalize_workflow(rf_wrkflw, rf_best)
rf_final_binary <- fit(rf_final_binary, training)

########################################

########################################
#           Multiclass Prep            #
########################################

load("results/rf-mutliclass.rda")
token_data <- nlp_fn_multi(parsed_claims)
token_data$mclass <- as.factor(token_data$mclass)

split <- initial_split(token_data, prop = 0.75, strata = mclass)
training <- training(split)
testing <- testing(split)
folds <- vfold_cv(data = token_data, v = 3, strata = mclass)

recipe <- recipe(mclass ~ ., data = training %>% select(-.id)) %>%
  step_pca(all_predictors(), num_comp = 100) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors(), -all_outcomes()) %>%
  step_nzv(all_predictors(), -all_outcomes())

rf_model <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = tune()
) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

rf_wrkflw <- workflow() %>%
  add_model(rf_model) %>%
  add_recipe(recipe)

rf_grid <- grid_random(
  mtry(range = c(1, 10)),
  trees(range = c(200, 400)),
  min_n(range = c(5, 10)),
  size = 20
)

rf_best <- select_best(rf_tune, metric = "roc_auc")
rf_final_multi <- finalize_workflow(rf_wrkflw, rf_best)
rf_final_multi <- fit(rf_final_multi, training)

########################################

# apply preprocessing pipeline
clean_df <- claims_test %>%
  parse_data() %>%
  nlp_fn_test()

train_columns <- colnames(training)
test_columns <- colnames(clean_df)
missing_columns <- setdiff(train_columns, test_columns)

missing_df <- as.data.frame(matrix(0, nrow = nrow(clean_df), ncol = length(missing_columns)))
colnames(missing_df) <- missing_columns
clean_df <- bind_cols(clean_df, missing_df)

clean_df <- clean_df[, train_columns]

# grab input
x <- clean_df %>%
  select(-.id, -bclass)

# compute predictions
preds_prob_binary <- predict(rf_final_binary, new_data = x, type = "prob")
colnames(preds_prob_binary) <- c("Not relevant", "Relevant")

preds_prob_multi <- predict(rf_final_multi, new_data = x, type = "prob")
colnames(preds_prob_multi) <- c("Not relevant", "Physical Activity", "Possible Fatality", "Potentially unlawful activity", "Other claim content")

preds_class_binary <- predict(rf_final_binary, new_data = x, type = "class")
colnames(preds_class_binary) <- c("Pred_class")

preds_class_multi <- predict(rf_final_multi, new_data = x, type = "class")
colnames(preds_class_multi) <- c("Pred_class")

binary_pred_df <- clean_df %>%
  select(.id) %>%
  bind_cols(preds_prob_binary, preds_class_binary)

multi_pred_df <- clean_df %>%
  select(.id) %>%
  bind_cols(preds_prob_multi, preds_class_multi)

# export (KEEP THIS FORMAT IDENTICAL)
# pred_df <- clean_df %>%
#   bind_cols(bclass.pred = pred_classes) %>%
#   select(.id, bclass.pred)

save(binary_pred_df, file = "results/binary-preds-group15.RData")
save(multi_pred_df, file = "results/multi-preds-group15.RData")
