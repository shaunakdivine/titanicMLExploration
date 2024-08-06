library(vip)

## This is all for cleaning up the loaded CSV ##
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

## This block groups titles used less than 5 times into 'Rare'
title_counts <- titanic_work %>%
  group_by(Title) %>%
  tally() %>%
  rename(Title_Count = n)

titanic_work <- titanic_work %>%
  left_join(title_counts, by = "Title") %>%
  mutate(Title = ifelse(Title_Count < 5, "Rare", Title)) %>%
  select(-Title_Count)

titanic_work$Title = factor(titanic_work$Title)

## Begin splitting and training here
set.seed(100)
train_ix = createDataPartition(titanic_work$Survived,
                               p = 0.8)

default_train = titanic_work[train_ix$Resample1,]
default_test  = titanic_work[-train_ix$Resample1,]

default_train <- na.omit(default_train)

## Set up CV using class code
kcv = 10
cv_folds = createFolds(default_train$Survived,
                       k = kcv)

my_summary = function(data, lev = NULL, model = NULL) {
  default = defaultSummary(data, lev, model)
  twoclass = twoClassSummary(data, lev, model)
  twoclass[3] = 1-twoclass[3]
  names(twoclass) = c("AUC_ROC", "TPR", "FPR")
  logloss = mnLogLoss(data, lev, model)
  c(default,twoclass, logloss)
}

fit_control <- trainControl(
  method = "cv",
  indexOut = cv_folds,
  classProbs = TRUE,
  savePredictions = TRUE,
  summaryFunction = my_summary,
  selectionFunction="oneSE")


###########################################################################
# Boosting
###########################################################################

gbm_grid <-  expand.grid(interaction.depth = c(3, 5, 10, 12), 
                         n.trees = c(100, 500, 750, 1000, 1500), 
                         shrinkage = c(0.1, .2),
                         n.minobsinnode = 10)

gbmfit <- train(Survived ~ ., data = default_train, 
                method = "gbm", 
                trControl = fit_control,
                tuneGrid = gbm_grid,
                metric = "roc_auc",
                verbose = FALSE)

print(gbmfit)
plot(gbmfit)

thresholder(gbmfit, 
            threshold = 0.7, 
            final = TRUE,
            statistics = c("Sensitivity",
                           "Specificity", "Precision", "Recall"))

gbmfit_res = thresholder(gbmfit, 
                         threshold = seq(0, 1, by = 0.01), 
                         final = TRUE)

optim_J = gbmfit_res[which.max(gbmfit_res$J),]

# ROC curve plot for training
ggplot(aes(x=1-Specificity, y=Sensitivity), data=gbmfit_res) + 
  geom_line() + 
  ylab("TPR (Sensitivity)") + 
  xlab("FPR (1-Specificity)") + 
  geom_abline(intercept=0, slope=1, linetype='dotted') +
  geom_segment(aes(x=1-Specificity, xend=1-Specificity, y=1-Specificity, yend=Sensitivity), color='darkred', data=optim_J) + 
  theme_bw()


best_pars = gbmfit$bestTune
best_preds = gbmfit$pred %>% filter(n.trees==best_pars$n.trees, 
                                    interaction.depth==best_pars$interaction.depth)

gbm_lift = caret::lift(pred~Survived, data=best_preds)

ggplot(gbm_lift) + 
  geom_abline(slope=1, linetype='dotted') +
  xlim(c(0, 10)) + 
  theme_bw()

# Calibration plot

gbm_cal = caret::calibration(obs~Survived, data=best_preds, cuts=7)
ggplot(gbm_cal) + theme_bw()

#### Holdout set results

test_probs = predict(gbmfit, newdata=default_test, type="prob")

get_metrics = function(threshold, test_probs, true_class, 
                       pos_label, neg_label) {
  # Get class predictions
  pc = factor(ifelse(test_probs[pos_label]>threshold, pos_label, neg_label), levels=c(pos_label, neg_label))
  test_set = data.frame(obs = true_class, pred = pc, test_probs)
  my_summary(test_set, lev=c(pos_label, neg_label))
}

# Get metrics for a given threshold
get_metrics(0.7, test_probs, default_test$Survived, "Survived", "Not_Survived")

# Compute metrics on test data using a grid of thresholds
thr_seq = seq(0, 1, length.out=500)
metrics = lapply(thr_seq, function(x) get_metrics(x, test_probs, default_test$Survived, "Survived", "Not_Survived"))
metrics_df = data.frame(do.call(rbind, metrics))

# ROC curve

ggplot(aes(x = FPR, y = TPR), data = metrics_df) + 
  geom_line(color = "blue", size = 1) +  # Color and size for the ROC curve
  geom_abline(intercept = 0, slope = 1, linetype = 'dotted', color = 'red') +  # Diagonal line
  labs(
    title = "ROC Curve",  # Title of the plot
    x = "FPR (1-Specificity)",  # X-axis label
    y = "TPR (Sensitivity)"  # Y-axis label
  ) +
  annotate(
    "text", x = 0.5, y = 0.3, 
    label = paste("AUC:", round(metrics_df$AUC_ROC[1], 2)),
    color = "darkred", size = 5, fontface = "bold"  # Customize AUC annotation
  ) +
  theme_minimal() +  # Clean theme
  theme(
    plot.title = element_text(hjust = 0.5),  # Center title
    axis.text = element_text(size = 12),  # Increase axis text size
    axis.title = element_text(size = 14),  # Increase axis title size
    panel.grid.major = element_line(color = "gray", linetype = "dotted")  # Add grid lines
  )

#Feature Importance
vip(gbmfit$finalModel, num_features = 10) + 
  ggtitle("Feature Importance for GBM Model") + 
  theme_minimal()

#Confusion Matrix
threshold <- 0.7
predicted_labels <- ifelse(test_probs$Survived > threshold, "Survived", "Not_Survived")
predicted_labels <- factor(predicted_labels, levels = levels(default_test$Survived))
conf_matrix <- confusionMatrix(predicted_labels, default_test$Survived)
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




