library(caret)
library(xgboost)
library(randomForest)
library(rpart)
library(rpart.plot)
library(tidyverse)
library(dplyr)
library(corrplot)
library(rvest)


head(TitanicData)#change the name download file
titanic = TitanicData
dim(titanic)

## Function to find null counts of variables
count_nulls_and_empty <- function(x) {
  sum(is.na(x) | x == "")
}

null_counts <- sapply(titanic, count_nulls_and_empty)
print(null_counts)


## Starting correlation matrix
numeric_data <- titanic %>% select_if(is.numeric)
correlation_matrix <- cor(numeric_data, use = "complete.obs")

print(correlation_matrix)
corrplot(correlation_matrix, method = "color", type = "upper", 
         order = "hclust", tl.col = "black", tl.srt = 45,
         addCoef.col = "black", number.cex = 0.4)

## Plot survival rate by passenger class
survival_rate_by_class <- titanic %>%
  group_by(Pclass) %>%
  summarize(
    Total = n(),
    Survived = sum(Survived, na.rm = TRUE),
    Survival_Rate = sum(Survived, na.rm = TRUE) / n()
  ) %>%
  na.omit()

print(survival_rate_by_class)
ggplot(survival_rate_by_class, aes(x = Pclass, y = Survival_Rate)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  theme_minimal() +
  labs(title = "Survival Rate by Passenger Class",
       x = "Passenger Class",
       y = "Survival Rate") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1))

## Plot survival rate by Sex
survival_rate_by_sex <- titanic %>%
  group_by(Sex) %>%
  summarize(
    Total = n(),
    Survived = sum(Survived, na.rm = TRUE),
    Survival_Rate = sum(Survived, na.rm = TRUE) / n()
  ) %>%
  na.omit()

print(survival_rate_by_sex)

ggplot(survival_rate_by_sex, aes(x = Sex, y = Survival_Rate)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  theme_minimal() +
  labs(title = "Survival Rate by Sex",
       x = "Sex",
       y = "Survival Rate") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1))


## Extract titles to make new column
extract_title <- function(name) {
  title <- str_extract(name, "(?<=,\\s)\\w+")
  return(title)
}

titanic <- titanic %>%
  mutate(Title = sapply(Name, extract_title))

#Had to fix one Title
titanic[760, "Title"] <- "Countess"


## Function to fill empty ages with median age for given Sex and Pclass
fill_age_with_median <- function(data) {
  data <- data %>%
    mutate(Age = ifelse(Age == "" | is.na(Age), NA, as.numeric(Age))) 
  
  median_age_by_group <- data %>%
    group_by(Sex, Pclass) %>%
    summarize(median_age = median(Age, na.rm = TRUE), .groups = 'drop')
  
  print(median_age_by_group)
  
  data <- data %>%
    left_join(median_age_by_group, by = c("Sex", "Pclass")) %>%
    mutate(Age = ifelse(is.na(Age), median_age, Age)) %>%
    select(-median_age)
  
  return(data)
}

titanic <- fill_age_with_median(titanic)

## Want to study age groups vs survival
calculate_survival_rate <- function(data, age_limit) {
  data %>%
    filter(Age < age_limit) %>%
    summarise(SurvivalRate = mean(Survived, na.rm = TRUE))
}

titanic$Survived <- as.numeric(as.character(titanic$Survived))
calculate_survival_rate_by_age_group <- function(data) {
  data %>%
    mutate(AgeGroup = cut(Age, breaks = seq(0, 80, by = 10), right = FALSE, include.lowest = TRUE)) %>%
    group_by(AgeGroup) %>%
    summarise(SurvivalRate = mean(Survived, na.rm = TRUE)) %>%
    na.omit()
}

age_group_survival_rates <- calculate_survival_rate_by_age_group(titanic)
print(age_group_survival_rates)
ggplot(age_group_survival_rates, aes(x = AgeGroup, y = SurvivalRate)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  theme_minimal() +
  labs(title = "Survival Rate by Age Group",
       x = "Age Group",
       y = "Survival Rate") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1))


## Want to make deck levels to try to gain something from cabin
extract_deck_level <- function(cabin) {
  if (!is.na(cabin) && nchar(cabin) > 0) {
    return(substr(cabin, 1, 1))
  } else {
    return(NA)
  }
}

titanic <- titanic %>%
  mutate(Deck_Level = sapply(Cabin, extract_deck_level))

## Finding average fare by deck
average_fare_by_deck <- titanic %>%
  group_by(Deck_Level) %>%
  summarise(Average_Fare = mean(Fare, na.rm = TRUE))

print(average_fare_by_deck)

## Want to fill in N/A rows with values where other passengers
## have identical price, likely same ticket or cabin
na_rows <- titanic %>%
  filter(is.na(Deck_Level))

known_rows <- titanic %>%
  filter(!is.na(Deck_Level))

num_changes <- 0

for (i in 1:nrow(na_rows)) {
  na_fare <- na_rows$Fare[i]
  
  matching_rows <- known_rows %>%
    filter(Fare == na_fare)
  
  if (nrow(matching_rows) > 0) {
    matching_deck_level <- matching_rows$Deck_Level[1]
    titanic$Deck_Level[is.na(titanic$Deck_Level) & titanic$Fare == na_fare] <- matching_deck_level
    num_changes <- num_changes + 1
  }
}

print(paste("Number of entries updated:", num_changes))


## Web scraping cabin data from titanic encyclopedia
url <- "https://www.encyclopedia-titanica.org/cabins.html"
webpage <- read_html(url)

cabins_data <- webpage %>%
  html_node("table") %>%
  html_table() %>%
  select(`Cabin No.`, Name) %>%
  rename(Cabin_No_Web = `Cabin No.`)

titanic_merge <- titanic %>%
  mutate(Name = tolower(Name)) %>%
  mutate(Name = gsub("\\.", "", Name))

cabins_data <- cabins_data %>%
  mutate(Name = tolower(Name)) %>%  
  mutate(Name = gsub("\\.", "", Name))

cabins_data_filtered <- cabins_data %>%
  anti_join(titanic_merge, by = "Name")

## Now by hand went thru and find index with value for titanic (issues with names/spelling)
#B86 271 Alexander Cairns
#C96 514 Rothschild
#C114 307 Fleming
#C112 381 Bidois
#C120 374 Ringhini
#C122 558 Robbins
#C138 538 LeRoy
#E161 157 Gilnagh

titanic[271, "Cabin"] <- "B 86"
titanic[514, "Cabin"] <- "C 96"
titanic[307, "Cabin"] <- "C 114"
titanic[381, "Cabin"] <- "C 112"
titanic[374, "Cabin"] <- "C 120"
titanic[558, "Cabin"] <- "C 122"
titanic[538, "Cabin"] <- "C 138"
titanic[157, "Cabin"] <- "E 161"

## Re-make deck level
titanic <- titanic %>%
  mutate(Deck_Level = sapply(Cabin, extract_deck_level))


# Re-doing Fare-Deck Level Extraction
known_rows <- titanic %>%
  filter(!is.na(Deck_Level))

na_rows <- titanic %>%
  filter(is.na(Deck_Level))

num_changes <- 0

for (i in 1:nrow(na_rows)) {
  na_fare <- na_rows$Fare[i]
  
  if (na_fare == 0) next
  
  matching_rows <- known_rows %>%
    filter(Fare == na_fare)
  
  if (nrow(matching_rows) > 0) {
    matching_deck_level <- matching_rows$Deck_Level[1]
    titanic$Deck_Level[is.na(titanic$Deck_Level) & titanic$Fare == na_fare] <- matching_deck_level
    num_changes <- num_changes + nrow(titanic[is.na(titanic$Deck_Level) & titanic$Fare == na_fare, ])
  }
}

print(paste("Number of entries updated:", num_changes))

## Assume deck level of 'steerage' for all 3rd class without cabin
## Assume deck level of 'midship' for all 2nd class without cabin
## This is information researched from Wikipedia about Titanic living
titanic$Deck_Level[is.na(titanic$Deck_Level) & titanic$Pclass == 3] <- 'S'
titanic$Deck_Level[is.na(titanic$Deck_Level) & titanic$Pclass == 2] <- 'M'


## Graph survival rate by deck
survival_rate_by_deck <- titanic %>%
  group_by(Deck_Level) %>%
  summarize(
    Total = n(),
    Survived = sum(Survived, na.rm = TRUE),
    Survival_Rate = sum(Survived, na.rm = TRUE) / n()
  ) %>%
  na.omit()

print(survival_rate_by_deck)
ggplot(survival_rate_by_deck, aes(x = Deck_Level, y = Survival_Rate)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  theme_minimal() +
  labs(title = "Survival Rate by Deck Level",
       x = "Deck Level",
       y = "Survival Rate") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1))


## Add a new column that sums SibSp and Parch into total_family
titanic <- titanic %>%
  mutate(total_family = SibSp + Parch)


titanic <- titanic %>%
  select(-PassengerId, -Name, -Cabin, -Ticket)
titanic <- titanic %>%
  select( -SibSp, -Parch)

