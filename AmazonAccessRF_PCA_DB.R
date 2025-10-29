#libaries
library(tidymodels)
library(embed)
library(vroom)
library(doParallel)
library(themis)


#setting up parallels
num_cores <- 4
cl <- makePSOCKcluster(num_cores)
registerDoParallel(cl)

#data import
trainData <- vroom::vroom("./train.csv") %>%
  mutate(ACTION = factor(ACTION))
testData <- vroom::vroom("./test.csv")

#recipe
amazon_recipe <- recipe(ACTION ~ ., data = trainData) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = 0.05) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors()) %>% 
  step_pca(all_predictors(), threshold = .9) %>% 
  step_smote(all_outcomes(), neighbors=2)

#prep the recipe
prepped_recipe <- prep(amazon_recipe)
final_predictor_count <- ncol(bake(prepped_recipe, new_data = NULL)) - 1

#model
amazon_model <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = 1000 
) %>% 
  set_engine('ranger', importance = "permutation", num.threads = num_cores) %>% 
  set_mode('classification')


#workflow
amazon_workflow <- workflow() %>%
  add_recipe(amazon_recipe) %>%
  add_model(amazon_model)

#number of cv splits
folds <- vfold_cv(trainData, v = 10, repeats = 1, strata = ACTION)

#tuning
latin_hypercube_grid <- grid_latin_hypercube(
  mtry(range = c(1L, final_predictor_count)), 
  min_n(range = c(2L, 40L)), 
  size = 30
)

#run cv
cv_results <- amazon_workflow %>%
  tune_grid(
    resamples = folds,
    grid = latin_hypercube_grid,
    metrics = metric_set(roc_auc),
    control = control_grid(verbose = TRUE) 
  )

#find best tuning params
best_tune <- cv_results %>%
  select_best(metric = 'roc_auc')

#final_wf
final_wf <- amazon_workflow %>%
  finalize_workflow(best_tune) %>%
  fit(data = trainData)

#predictions
submission_predictions <- final_wf %>%
  predict(new_data = testData, type = 'prob') %>%
  bind_cols(testData) %>%
  rename(ACTION = .pred_1) %>%
  select(id, ACTION)

#saving output
vroom::vroom_write(submission_predictions, "submissionRF.csv", delim = ",")

stopCluster(cl)

#Server Commands (MAKE SURE TO BE IN RIGHT FILE)
# ssh bjm259@stat-u02.byu.edu
# R CMD BATCH --no-save --no-restore AmazonAccessRF_PCA_DB.R & 
# top <- to see if it is running
# less AmazonAccess.Rout <- to see the Rout file
