library(MASS)
library(caret)
library(randomForest)
library(gbm)
library(rpart)
library(rpart.plot)
library(tidyverse)
library(dplyr)
library(ISLR2)
library(MLmetrics)
library(MLeval)
library(pROC)
library(vip)

# clean the data from titanic_cleaned_data
# create factors
titanic_work <- titanic_cleaned_new
titanic_work <- titanic_work[-c(62, 830), ]
str(titanic_work)
titanic_work <- na.omit(titanic_work)
titanic_work <- titanic_work %>% select(-X)
titanic_work$Survived = factor(titanic_work$Survived, levels = c("1", "0"),
                               labels = c("Survived", "Not_Survived"))
titanic_work$Pclass = factor(titanic_work$Pclass, levels = c("1", "2", "3"),
                             labels = c("Class1", "Class2", "Class3"))
titanic_work$Sex = factor(titanic_work$Sex)
titanic_work$Embarked = factor(titanic_work$Embarked)
titanic_work$Deck_Level = factor(titanic_work$Deck_Level)
title_counts <- titanic_work %>%
  group_by(Title) %>%
  tally() %>%
  rename(Title_Count = n)
# Combine titles with fewer than 5 entries into "Rare"
titanic_work <- titanic_work %>%
  left_join(title_counts, by = "Title") %>%
  mutate(Title = ifelse(Title_Count < 5, "Rare", Title)) %>%
  select(-Title_Count)  # Remove the count column if no longer needed
titanic_work$Title = factor(titanic_work$Title)

set.seed(100) # create data partitions for the model
train_ix = createDataPartition(titanic_work$Survived,
                               p = 0.8)

titanic_train = titanic_work[train_ix$Resample1,]
titanic_test  = titanic_work[-train_ix$Resample1,]



my_summary = function(data, lev = NULL, model = NULL) {
  default = defaultSummary(data, lev, model)
  twoclass = twoClassSummary(data, lev, model)
  # Converting to TPR and FPR instead of sens/spec
  twoclass[3] = 1-twoclass[3]
  names(twoclass) = c("AUC_ROC", "TPR", "FPR")
  logloss = mnLogLoss(data, lev, model)
  c(default,twoclass, logloss)
}

fit_control <- trainControl(
  method = "cv",
  # Save predicted probabilities, not just classifications
  classProbs = TRUE,
  # Save all the holdout predictions, to summarize and plot
  savePredictions = TRUE,
  summaryFunction = my_summary,
  selectionFunction="oneSE")

# picked nbagg = 1000 because accuracy only marginally got better
model <- train(Survived ~ .,
               data = titanic_train,
               method = "treebag",
               trControl = fit_control,
               nbagg = 1000)
print(model)

test_probs <- predict(model, titanic_test, type = 'prob')


#Confusion Matrix
# Generate predicted class labels based on a threshold
threshold <- 0.5035

predicted_labels <- ifelse(test_probs$Survived > threshold, "Survived", "Not_Survived")
# Convert to factor with the same levels as the true labels
predicted_labels <- factor(predicted_labels, levels = levels(titanic_test$Survived))
# Generate the confusion matrix
conf_matrix <- confusionMatrix(predicted_labels, titanic_test$Survived)
# Print the confusion matrix
print(conf_matrix)
cm <- as.table(conf_matrix$table)
df_cm <- as.data.frame(cm)
# Plot the confusion matrix
ggplot(data = df_cm, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "white", high = "steelblue") +
  geom_text(aes(label = sprintf("%d", Freq)), vjust = 1) +
  theme_minimal() +
  labs(title = "Confusion Matrix", x = "Actual", y = "Predicted") +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title = element_text(hjust = 0.5)
  )

# Generate ROC curve
roc_curve <- roc(titanic_test$Survived, test_probs$Survived)

# Plot ROC curve using ggplot2
roc_data <- data.frame(
  TPR = roc_curve$sensitivities,
  FPR = 1 - roc_curve$specificities
)

ggplot(roc_data, aes(x = FPR, y = TPR)) +
  geom_line(color = "blue", size = 1) +
  geom_abline(linetype = "dashed", color = "red") +
  labs(
    title = "ROC Curve for Titanic Survival Prediction",
    x = "False Positive Rate",
    y = "True Positive Rate"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    axis.text = element_text(size = 12),
    axis.title = element_text(size = 14)
  )

#Feature Importance
importance <- varImp(model, scale = FALSE)
print(importance)

# Plot feature importance
vip(model, num_features = 10) +
  ggtitle("Feature Importance for Treebag Model") +
  theme_minimal()

