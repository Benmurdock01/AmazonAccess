#libaries
library(tidymodels)
library(embed)
library(vroom)
library(doParallel)
library(discrim)

#setting up parallels
num_cores <- 6
cl <- makePSOCKcluster(num_cores)
registerDoParallel(cl)

#data import
trainData <- vroom::vroom("./train.csv") %>%
  mutate(ACTION = factor(ACTION))
testData <- vroom::vroom("./test.csv")

#recipe
amazon_recipe <- recipe(ACTION ~ ., data = trainData) %>%
  step_mutate(across(all_predictors(), as.factor)) %>%
  step_other(all_nominal_predictors(), threshold = 0.05) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_zv(all_predictors())

#model
amazon_model <- naive_Bayes(Laplace = tune(), smoothness = tune()) %>% 
  set_mode('classification') %>% 
  set_engine('naivebayes')

#workflow
amazon_workflow <- workflow() %>%
  add_recipe(amazon_recipe) %>%
  add_model(amazon_model)

#number of cv splits
folds <- vfold_cv(trainData, v = 10, repeats = 1, strata = ACTION)

#tuning
tuning_grid <- grid_regular(
  smoothness(range = c(0.5, 2.0)),
  Laplace(),                   
  levels = 5                  
)

#run cv
cv_results <- amazon_workflow %>%
  tune_grid(
    resamples = folds,
    grid = tuning_grid,
    metrics = metric_set(roc_auc)
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
vroom::vroom_write(submission_predictions, "submissionNavie.csv", delim = ",")

stopCluster(cl)

#Server Commands (MAKE SURE TO BE IN RIGHT FILE)
# ssh bjm259@stat-u02.byu.edu
# R CMD BATCH --no-save --no-restore AmazonAccess.R & 
# top <- to see if it is running
# less AmazonAccess.Rout <- to see the Rout file
