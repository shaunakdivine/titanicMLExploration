library(caret)
library(tidyverse)
library(ggplot2)
library(dplyr)
library(randomForest)
library(pROC)

set.seed(100)

# Changing variable name to write code easier
data  <- titanic_work

# Scaling variables
preprocessParams <- preProcess(data, method = c("center", "scale"))
data <- predict(preprocessParams, data)

# Splitting data into training and testing subsets
trainIndex <- createDataPartition(data$Survived, p = 0.8, list = FALSE)
trainData <- data[trainIndex,]
testData <- data[-trainIndex,]

# Creating knn model with bootstrap resampling
knnFit1 <- train(Survived ~ ., 
                 data = trainData, 
                 method = "knn",
                 tuneLength = 10)

# Applying our model to the testData
predictions1 <- predict(knnFit1, newdata = testData)

# Building a confusion matrix
cm1 <- confusionMatrix(predictions1, testData$Survived)

# Extract confusion matrix data for the first predictions
df_cm1 <- as.data.frame(cm1$table)

# Now, let's build a ROC graph of model
probabilities1 <- predict(knnFit1, newdata = testData, type = "prob")

# Calculate the ROC curve
roc_curve <- roc(testData$Survived, probabilities1[, "Survived"])

# Calculate the AUC
auc_value <- auc(roc_curve)
cat("AUC is", auc_value, "\n")

# Calculate the optimal threshold using Youden's J statistic
optimal_threshold <- coords(roc_curve, "best", ret = "threshold", best.method = "youden")
cat("Optimal Threshold:", optimal_threshold[,1], "\n")

optimal_threshold <- optimal_threshold[[1]]

predicted_probabilities <- probabilities1[, "Survived"]

# Apply the optimal threshold to make class predictions
new_predictions <- ifelse(predicted_probabilities >= optimal_threshold, "Survived", "Not_Survived")

# Convert to factors with the same levels as the original data
new_predictions <- factor(new_predictions, levels = levels(testData$Survived))

# Evaluate the model using the new predictions
cm2 <- confusionMatrix(new_predictions, testData$Survived)

# Extract confusion matrix data for the new predictions
df_cm2 <- as.data.frame(cm2$table)

# Plot confusion matrix for the first predictions
ggplot(data = df_cm1, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "white", high = "steelblue") +
  geom_text(aes(label = sprintf("%d", Freq)), vjust = 1) +
  theme_minimal() +
  labs(title = "Confusion Matrix - Initial Predictions", x = "Actual", y = "Predicted") +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title = element_text(hjust = 0.5)
  )

# Plot confusion matrix for the new predictions
ggplot(data = df_cm2, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "white", high = "steelblue") +
  geom_text(aes(label = sprintf("%d", Freq)), vjust = 1) +
  theme_minimal() +
  labs(title = "Confusion Matrix - Threshold Optimized Predictions", x = "Actual", y = "Predicted") +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title = element_text(hjust = 0.5)
  )

# creating a data frame of specificity and sensitivity 
roc_df <- data.frame(
  specificity = roc_curve$specificities,
  sensitivity = roc_curve$sensitivities
)

# Plot ROC curve with the threshold level
ggplot(roc_df, aes(x = 1 - specificity, y = sensitivity)) +
  geom_line(color = "blue") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") + 
  geom_vline(xintercept = optimal_threshold, color = "red", linetype = "dashed") +
  annotate("text", x = 0.8, y = 0.1, label = paste("AUC =", round(auc_value, 3)), color = "black") +
  labs(title = "ROC Curve for KNN Model", x = "1 - Specificity (False Positive Rate)", y = "Sensitivity (True Positive Rate)") +
  theme_minimal()

