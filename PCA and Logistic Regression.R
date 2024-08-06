library(caret)
library(xgboost)
library(randomForest)
library(rpart)
library(rpart.plot)
library(tidyverse)
library(dplyr)
library(corrplot)
library(rvest)
library(ggplot2)
library(devtools)
library(ggbiplot)
library(e1071)
library(gbm)
library(pROC)

#PCA: Principal Components Analysis
titanic = read.csv("titanic_cleaned_new.csv")

#Select Only Numeric Columns and save as a new file
selected_columns <- titanic [,c("Survived", "Pclass", "Age", "Fare", "total_family")]

#write to new csv file
write.csv(selected_columns, "C:/Users/krgod/Documents/Texas MSBA/Summer Semester/STA 380/titanic_pca_file.csv", row.names = FALSE)

#standardize data
PreProcessedValues <- preProcess(selected_columns, method = c("center", "scale"))
titanic_pca <- predict(PreProcessedValues, selected_columns)

#PCA Analysis

#list of Principal Components
#PC1 --> Survived
#PC2 --> Pclass
#PC3 --> Age
#PC4 --> Fare
#PC5 --> total_Family


pca_result <- prcomp(titanic_pca, center = TRUE, scale. = TRUE)
summary(pca_result)

pca_data <- as.data.frame(pca_result$x)
pca_data$Survived <- selected_columns$Survived

ggplot(pca_data, aes(x = PC2, y = PC3, color = Survived)) +
  geom_point(size = 2) + 
  labs(title = "PCA of Titanic Survival", 
       x = "Principal Component 2",
       y = "Principal Component 3")

pca_result <- prcomp(titanic_pca, center = TRUE, scale. = TRUE)
summary(pca_result)

pca_data <- as.data.frame(pca_result$x)
pca_data$Survived <- selected_columns$Survived

ggplot(pca_data, aes(x = PC2, y = PC4, color = Survived)) +
  geom_point() + 
  scale_x_continuous(limits = c(0,3), breaks = seq(0,2, by = 1), labels = seq(0,2, by = 1))+
  scale_y_continuous(limits = c(0,300), breaks = seq(0,300, by = 50), labels = seq(0,300, by = 50))+
  labs(title = "PCA of Titanic Survival", 
       x = "Pclass",
       y = "Fare")








#PCA Using Caret
train_control <- trainControl(method = "cv", number = 10)

model <- train(Survived ~ ., data = titanic,
               method = "knn", 
               preProcess = c("center", "scale", "pca"),
               trControl = train_control)


print(model)




#Logistic Regression
titanic_cleaned_new = read.csv("C:/Users/krgod/Documents/Texas MSBA/Summer Semester/STA 380/titanic_cleaned_new.csv")

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
# titanic$Survived <- as.factor(titanic$Survived)

set.seed(100)
trainIndex <- createDataPartition(titanic_work$Survived , p = 0.80,
                                  list = FALSE,
                                  times = 1)
dataTrain <- titanic_work[trainIndex, ]
dataTest <- titanic_work[-trainIndex, ]
model <- train(Survived ~ ., data = dataTrain, method = "glm", family = binomial)
print(model)
prediction_prob <- predict(model, newdata = dataTest, type = "prob")[, "Survived"]
binary_prediction <- ifelse(prediction_prob > 0.50, "Survived", "Not_Survived")
binary_prediction <- factor(binary_prediction, levels = levels(dataTest$Survived))
confusion <- confusionMatrix(binary_prediction, dataTest$Survived)
print(confusion)
very_important_model <-model

#Useful Plots with Logistic Regression Model

#Coefficient Plot
coefficients <- as.data.frame(summary(model)$coef)
coefficients$Variable <- rownames(coefficients)
coefficients <- coefficients[order(coefficients$Estimate), ]

ggplot(coefficients, aes(x = reorder(Variable, Estimate), y = Estimate)) +
  geom_point() +
  coord_flip() +
  ggtitle("Logistic Regression Coefficients") +
  xlab("Variable") +
  ylab("Coefficient Estimate")

# ROC Curve
roc_obj <- roc(dataTest$Survived, prediction_prob, levels = rev(levels(dataTest$Survived)))
plot(roc_obj, main = "ROC Curve", col = "blue", lwd = 2)
abline(a = 0, b = 1, col = "red", lty = 2)
auc_val <- auc(roc_obj)
text(0.6, 0.2, paste("AUC =", round(auc_val, 2)), col = "blue")

#Confusion Matrix 
confusion_df <- as.data.frame(confusion$table)
ggplot(data = confusion_df, mapping = aes(x = Prediction, y = Reference)) +
  geom_tile(aes(fill = Freq)) +
  geom_text(aes(label = Freq)) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  ggtitle("Confusion Matrix") +
  xlab("Predicted") +
  ylab("Actual")

# Actual vs Predicted Outcomes Plot
dataTest$Predicted_Probabilities <- prediction
ggplot(dataTest, aes(x = Predicted_Probabilities, fill = Survived)) +
  geom_histogram(binwidth = 0.05, alpha = 0.5, position = "identity") +
  ggtitle("Predicted Probabilities vs. Actual Outcomes") +
  xlab("Predicted Probability of Survival") +
  ylab("Frequency") +
  scale_fill_manual(values = c("Survived" = "green", "Not_Survived" = "red"))










