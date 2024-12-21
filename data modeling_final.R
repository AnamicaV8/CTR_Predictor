library(caret)
library(pROC)
#-------------------------------Logistic Regression-----------------------------------
# Load necessary libraries
library(pscl)
library(MASS)

# Load data
data <- read.csv("C:/Users/junya/Desktop/Data Science for Business/final project/shrinked_data_V2.csv")

# Split data into train and test sets
sample_index <- sample(1:nrow(data), size = 0.7 * nrow(data))
train_data <- data[sample_index, ]
test_data <- data[-sample_index, ]

# Ensure the label is a factor with two levels for classification
train_data$label <- as.factor(train_data$label)
test_data$label <- as.factor(test_data$label)

# Make sure factor levels are valid R variable names
levels(train_data$label) <- make.names(levels(train_data$label))
levels(test_data$label) <- make.names(levels(test_data$label))

# Define the control for cross-validation (5-fold)
train_control_lr <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)

# Stepwise Logistic Regression using stepAIC
# Start with a full model (all variables)
full_model_lr <- glm(label ~ ., data = train_data, family = "binomial")

# Stepwise selection to find the best model
stepwise_model_lr <- stepAIC(full_model_lr, direction = "both", trace = FALSE)

# Print the selected variables in the best model
summary(stepwise_model_lr)

# Train logistic regression model with selected variables using 5-fold cross-validation
set.seed(123)
logit_model_cv_lr <- train(
  formula(stepwise_model_lr), 
  data = train_data,
  method = "glm",
  family = "binomial",
  trControl = train_control_lr,
  metric = "ROC"
)

# Print the cross-validation results
print(logit_model_cv_lr)

# Evaluate the final model on the test dataset
# Predict probabilities for the positive class (label = '1')
pred_prob_test_lr <- predict(logit_model_cv_lr, newdata = test_data, type = "prob")[, 2]

# Generate the ROC curve and calculate AUC on the test data
roc_curve_lr <- roc(test_data$label, pred_prob_test_lr)
auc_value_lr <- auc(roc_curve_lr) # AUC=0.6223
print(paste("AUC on Test Data:", round(auc_value_lr, 4)))

# Plot the ROC curve
plot(roc_curve_lr, main = "ROC Curve for Logistic Regression (Test Data)", col = "#1c61b6", lwd = 2)
abline(a = 1, b = -1, col = "red", lty = 2)  # Random guessing line
legend("bottomright", legend = paste("AUC =", round(auc_value_lr, 4)), col = "#1c61b6", lwd = 2)
#--------------------------------CART-------------------------------------------------
library(dplyr)
library(rpart)

# Undersample the majority class
majority_class <- train_data %>% filter(label == "X0")
minority_class <- train_data %>% filter(label == "X1")
set.seed(123)
undersampled_majority <- majority_class %>% sample_n(nrow(minority_class))  # Undersample to match the minority class size
balanced_data <- bind_rows(minority_class, undersampled_majority) # balanced_train_data

# initial model
cart_model_initial <- rpart(label ~ creat_type_cd + inter_type_cd + slot_id + age + city_rank + net_type + communication_avgonline_30d + 
                        indu_name + device_size + list_time, 
                      data = balanced_data, 
                      method = "class",  # Ensure it's a classification tree
                      control = rpart.control(cp = 0.001, minsplit = 10, maxdepth = 15))
# Predict probabilities on the test set for class '1'
pred_prob1 <- predict(cart_model_initial, test_data, type = "prob")[, 2]
# Generate the ROC curve
roc_curve1 <- roc(test_data$label, pred_prob1)
# Plot the ROC curve
plot(roc_curve1, main = "ROC Curve for CART Model", col = "#1c61b6", lwd = 2)
abline(a = 0, b = 1, col = "red", lty = 2)  # Random guessing line
legend("bottomright", legend = paste("AUC =", round(auc(roc_curve1), 4)), col = "#1c61b6", lwd = 2)
# Print the AUC value
auc_value1 <- auc(roc_curve1)
print(paste("AUC:", round(auc_value1, 4)))

# Feature transformations
balanced_data$app_interaction <- balanced_data$app_first_class * balanced_data$app_second_class
test_data$app_interaction <- test_data$app_first_class * test_data$app_second_class

balanced_data$age_career_interaction <- balanced_data$age * balanced_data$career
test_data$age_career_interaction <- test_data$age * test_data$career

balanced_data$creat_inter_interaction <- balanced_data$creat_type_cd * balanced_data$inter_type_cd
test_data$creat_inter_interaction <- test_data$creat_type_cd * test_data$inter_type_cd

balanced_data$log_device_price <- log(balanced_data$device_price + 1)
test_data$log_device_price <- log(test_data$device_price + 1)

#Define Cross-Validation Method
train_control_stratified <- trainControl(
  method = "cv",                   # Cross-validation
  number = 10,                     # 10 folds
  classProbs = TRUE,               # Use class probabilities
  summaryFunction = twoClassSummary  # Calculate ROC AUC
)

# Define Hyperparameter Grid for Tuning
tune_grid <- expand.grid(cp = seq(0.0001, 0.01, by = 0.001))

# Train CART Model Using Stratified Cross-Validation
set.seed(123)  # For reproducibility
cart_tuned <- train(
  label ~ creat_inter_interaction + slot_id + indu_name + device_size + log_device_price + app_interaction + age_career_interaction, 
  data = balanced_data, 
  method = "rpart",
  metric = "ROC",  # Use ROC to find the best model
  trControl = train_control_stratified,
  tuneGrid = tune_grid
)

# Check model results
print(cart_tuned)

# View the best `cp` parameter found
print(cart_tuned$bestTune)

# Train Final CART Model Using Best Parameters
final_cart_model <- rpart(
  label ~ creat_inter_interaction + slot_id + indu_name + device_size + log_device_price + app_interaction + age_career_interaction, 
  data = balanced_data, 
  method = "class",
  control = rpart.control(cp = cart_tuned$bestTune$cp, minsplit = 10, maxdepth = 15)  # Use the best cp
)

# Evaluate the Final Model on Test Data
# Predict probabilities for the positive class (label = '1')
pred_prob_final <- predict(final_cart_model, test_data, type = "prob")[, 2]

# Generate the ROC curve and calculate AUC
roc_curve_final <- roc(test_data$label, pred_prob_final)
auc_value_final <- auc(roc_curve_final)
print(paste("AUC of Final Model:", round(auc_value_final, 4)))

# Plot the ROC curve for the final model
plot(roc_curve_final, main = "ROC Curve for Improved CART Model", col = "#1c61b6", lwd = 2)
abline(a = 1, b = -1, col = "red", lty = 2)  # Random guessing line
legend("bottomright", legend = paste("AUC =", round(auc_value_final, 4)), col = "#1c61b6", lwd = 2)

# Prune the Final Model (If Needed)
# Prune the tree using the optimal cp
pruned_tree <- prune(final_cart_model, cp = cart_tuned$bestTune$cp)

# Evaluate the pruned tree's performance
pred_prob_pruned <- predict(pruned_tree, test_data, type = "prob")[, 2]
roc_curve_pruned <- roc(test_data$label, pred_prob_pruned)
auc_value_pruned <- auc(roc_curve_pruned)
print(paste("AUC after Pruning:", round(auc_value_pruned, 4)))

# Plot ROC curve for pruned model
plot(roc_curve_pruned, main = "ROC Curve for Pruned CART Model", col = "#1c61b6", lwd = 2)
abline(a = 1, b = -1, col = "red", lty = 2)
legend("bottomright", legend = paste("AUC =", round(auc_value_pruned, 4)), col = "#1c61b6", lwd = 2)

#---------------------------------Random Forest-------------------------------------------------------
library(randomForest)

# Undersampling the majority class (label = 0)
set.seed(123)
undersampled_data <- train_data[which(train_data$label == "X1"), ] # Keep all minority class (label = 1)
majority_class <- train_data[which(train_data$label == "X0"), ]
undersample_indices <- sample(1:nrow(majority_class), size = nrow(undersampled_data))
undersampled_data <- rbind(undersampled_data, majority_class[undersample_indices, ])

# Define training control with 5-fold cross-validation
train_control <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)

# Define the grid of `mtry` values to search over (you can adjust the range based on your dataset size)
tune_grid <- expand.grid(mtry = seq(1, ncol(undersampled_data) - 1, by = 1))  # Tuning mtry from 1 to the number of features

# Train the Random Forest model with grid search for `mtry`
set.seed(123)
rf_model <- train(label ~ ., 
                  data = undersampled_data, 
                  method = "rf", 
                  metric = "ROC", 
                  tuneGrid = tune_grid, 
                  trControl = train_control)

# Print the optimal mtry value
rf_model$bestTune #mtry = 2

# Define the grid with mtry = 2 for Random Forest
rf_grid_best <- expand.grid(mtry = 2)

# Train the Random Forest model using 5-fold cross-validation
set.seed(123)
rf_model_best <- train(
  label ~ ., 
  data = undersampled_data, 
  method = "rf",  # Random Forest
  trControl = train_control, 
  tuneGrid = rf_grid_best,  # mtry = 2
  metric = "ROC"  # Use ROC-AUC as the performance metric
)

# Generate ROC curve using cross-validated predictions
rf_pred_prob <- predict(rf_model_best, test_data, type = "prob")[,2]  # Predicted probabilities for class 1
roc_curve <- roc(test_data$label, rf_pred_prob)
plot(roc_curve, col = "blue", main = "ROC Curve (Random Forest with mtry=2)")
abline(a = 1, b = -1, col = "red", lty = 2)
legend("bottomright", legend = paste("AUC =", round(auc(roc_curve), 4)), col = "#1c61b6", lwd = 2)
# Print ROC-AUC value
print(paste("ROC-AUC:", round(auc(roc_curve), 4)))
# if used to predict the test data, ROC-AUC: 0.671

#-----------------------------XGBoost------------------------------------------
# Load necessary libraries
library(xgboost)
install.packages("doParallel")
library(doParallel)

# Register multiple cores for parallel processing
cl <- makePSOCKcluster(detectCores() - 1)  # Use all but one core for processing
registerDoParallel(cl)

# Set up train control with 5-fold cross-validation
train_control <- trainControl(method = "cv", 
                              number = 5, 
                              classProbs = TRUE, 
                              summaryFunction = twoClassSummary,  # To calculate ROC-AUC
                              savePredictions = TRUE)  # Save predictions to calculate OOS R-squared

# Function to train XGBoost with specific hyperparameters and output ROC-AUC
train_xgboost <- function(eta, max_depth, gamma, colsample_bytree, min_child_weight, subsample) {
  # Define the hyperparameters manually
  xgb_params <- expand.grid(
    nrounds = 100,  # Fixed number of boosting iterations
    eta = eta,  # Learning rate
    max_depth = max_depth,  # Maximum tree depth
    gamma = gamma,  # Minimum loss reduction to make a split
    colsample_bytree = colsample_bytree,  # Fraction of columns sampled for each tree
    min_child_weight = min_child_weight,  # Minimum sum of instance weight in a child
    subsample = subsample  # Subsample ratio of training instances
  )
  
  # Train the XGBoost model
  set.seed(123)
  xgb_model <- train(
    label ~ ., 
    data = data, 
    method = "xgbTree", 
    trControl = train_control, 
    tuneGrid = xgb_params, 
    metric = "ROC"  # Use ROC-AUC as the performance metric
  )
  
  # Output the best ROC-AUC score
  round(max(xgb_model$results$ROC), 4)
}

# Define 5 sets of hyperparameters manually
hyperparameters <- list(
  list(eta = 0.01, max_depth = 3, gamma = 0, colsample_bytree = 0.6, min_child_weight = 1, subsample = 0.7),
  list(eta = 0.05, max_depth = 5, gamma = 1, colsample_bytree = 0.8, min_child_weight = 2, subsample = 0.8),
  list(eta = 0.1, max_depth = 4, gamma = 2, colsample_bytree = 0.7, min_child_weight = 3, subsample = 0.85),
  list(eta = 0.15, max_depth = 6, gamma = 3, colsample_bytree = 0.9, min_child_weight = 4, subsample = 0.9),
  list(eta = 0.2, max_depth = 9, gamma = 5, colsample_bytree = 1, min_child_weight = 5, subsample = 0.75)
)

# Run the model 5 times, each with a different set of hyperparameters
results <- lapply(hyperparameters, function(params) {
  train_xgboost(params$eta, params$max_depth, params$gamma, params$colsample_bytree, params$min_child_weight, params$subsample)
})

# Print the results of each run
for (i in 1:5) {
  cat("Run", i, "\n")
  cat("Best ROC-AUC:", results[[i]], "\n")
  cat("\n")
}

# Run 1 
# Best ROC-AUC: 0.6566 
# Run 2 
# Best ROC-AUC: 0.6821 
# Run 3 
# Best ROC-AUC: 0.6865 
# Run 4 
# Best ROC-AUC: 0.6893 
# Run 5 
# Best ROC-AUC: 0.68

# Define 5 new sets of hyperparameters manually based on previous results
new_hyperparameters <- list(
  list(eta = 0.12, max_depth = 6, gamma = 3, colsample_bytree = 0.85, min_child_weight = 4, subsample = 0.88),  # Fine-tuning based on best performance
  list(eta = 0.08, max_depth = 7, gamma = 4, colsample_bytree = 0.9, min_child_weight = 5, subsample = 0.86),   # Increase gamma and adjust depth
  list(eta = 0.1, max_depth = 5, gamma = 2.5, colsample_bytree = 0.87, min_child_weight = 3, subsample = 0.9),  # Balance between learning rate and depth
  list(eta = 0.09, max_depth = 6, gamma = 2, colsample_bytree = 0.95, min_child_weight = 3, subsample = 0.85),  # High colsample with moderate depth
  list(eta = 0.11, max_depth = 8, gamma = 3.5, colsample_bytree = 0.8, min_child_weight = 4, subsample = 0.83)  # Increase depth and gamma
)

# Run the model 5 times, each with a different set of new hyperparameters
new_results <- lapply(new_hyperparameters, function(params) {
  train_xgboost(params$eta, params$max_depth, params$gamma, params$colsample_bytree, params$min_child_weight, params$subsample)
})

# Print the ROC-AUC results of each new run
for (i in 1:5) {
  cat("New Run", i, "\n")
  cat("Best ROC-AUC:", new_results[[i]], "\n")
  cat("\n")
}

# New Run 1 
# Best ROC-AUC: 0.6892 
# New Run 2 
# Best ROC-AUC: 0.6892 
# New Run 3 
# Best ROC-AUC: 0.6885 
# New Run 4 
# Best ROC-AUC: 0.6894 
# New Run 5 
# Best ROC-AUC: 0.688

#fit the best model and generate ROC curve
# Set up the train control for 5-fold cross-validation
train_control <- trainControl(method = "cv", 
                              number = 5, 
                              classProbs = TRUE, 
                              summaryFunction = twoClassSummary,  # To calculate ROC-AUC
                              savePredictions = TRUE)

# Define the specific hyperparameters you want to use
xgb_params <- expand.grid(
  nrounds = 100,  # Fixed number of boosting iterations
  eta = 0.09,  # Learning rate
  max_depth = 6,  # Maximum tree depth
  gamma = 2,  # Minimum loss reduction to make a split
  colsample_bytree = 0.95,  # Fraction of columns sampled for each tree
  min_child_weight = 3,  # Minimum sum of instance weight in a child
  subsample = 0.85  # Subsample ratio of training instances
)

# Train the XGBoost model using the specified hyperparameters
set.seed(123)
xgb_model <- train(
  label ~ ., 
  data = undersampled_data, 
  method = "xgbTree", 
  trControl = train_control, 
  tuneGrid = xgb_params, 
  metric = "ROC"  # Use ROC-AUC as the performance metric
)

# Generate predictions on the dataset using the trained model
xgb_pred_prob <- predict(xgb_model, test_data, type = "prob")[,2]  # Predicted probabilities for class 1

# Calculate the ROC-AUC and plot the ROC curve
roc_curve_xgb <- roc(test_data$label, xgb_pred_prob)
plot(roc_curve_xgb, col = "blue", main = "ROC Curve (XGBoost Model)")
abline(a = 1, b = -1, col = "red", lty = 2)
legend("bottomright", legend = paste("AUC =", round(auc(roc_curve_xgb), 4)), col = "blue", lwd = 2)

# Print ROC-AUC value
print(paste("ROC-AUC:", round(auc(roc_curve_xgb), 4)))
# if used to predict the whole shrinked data, ROC-AUC: 0.6804

# Stop the parallel cluster
stopCluster(cl)
# Return to sequential processing mode
registerDoSEQ()