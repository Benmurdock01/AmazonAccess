#libaries
library(tidymodels)
library(embed)
library(vroom)
library(doParallel)


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
  step_mutate_at(all_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = 0.05) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold = .7)

amazon_model <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>%
  set_mode('classification') %>%
  set_engine('kernlab', maxit = 100000)

#workflow
amazon_workflow <- workflow() %>%
  add_recipe(amazon_recipe) %>%
  add_model(amazon_model)

#tuning
tuning_grid <- expand_grid(
  cost = 10^seq(-3, 1, length.out = 5),
  rbf_sigma = 10^seq(-2, 0, length.out = 5)
)

#number of cv splits
folds <- vfold_cv(trainData, v = 10)

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
final_fit <- amazon_workflow %>%
  finalize_workflow(best_tune) %>%
  fit(data = trainData)

#predictions
submission_predictions <- final_fit %>% # Use 'final_fit' here
  predict(new_data = testData, type = 'prob') %>%
  bind_cols(testData) %>%
  rename(ACTION = .pred_1) %>%
  select(id, ACTION)

#saving output
vroom::vroom_write(submission_predictions, "submissionSVM_PCA.csv", delim = ",")

stopCluster(cl)

#Server Commands (MAKE SURE TO BE IN RIGHT FILE)
# ssh bjm259@stat-u02.byu.edu
# R CMD BATCH --no-save --no-restore AmazonAccessSVM_PCA.R &
# top <- to see if it is running
# less AmazonAccessKNN.Rout <- to see the Rout file
