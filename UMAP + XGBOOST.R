library(umap)
library(data.table)
library(caret)

##########################################################################################
# DATA PREPROCESSING
##########################################################################################

train_file_path <- "train.csv"
test_file_path <- "test.csv"

train_df <- fread(train_file_path)
test_df <- fread(test_file_path)


month_map <- c("Jan" = 1, "Feb" = 2, "Mar" = 3, "Apr" = 4, "May" = 5, "Jun" = 6, 
               "Jul" = 7, "Aug" = 8, "Sep" = 9, "Oct" = 10, "Nov" = 11, "Dec" = 12)
day_map   <- c("Morning" = 0, "Afternoon" = 1, "Night" = 2)

dummy <- dummyVars("~ y", data = train_df)
train_df_transformed <- data.table(predict(dummy, newdata = train_df))
train_df <- cbind(train_df[, !"y", with = FALSE], train_df_transformed)

train_df[, month := month_map[month]]
train_df[, day := day_map[day]]


test_df[, month := month_map[month]]
test_df[, day := day_map[day]]

train_df[is.na(train_df)] <- 0
test_df[is.na(test_df)] <- 0

##########################################################################################
# Dimensionality reduction
##########################################################################################
train_df_umap <- train_df[, .SD, .SDcols = !c("id", "yZ0", "yZ1", "yZ2", "yZ3", "yZ4", "yZ5")]
test_df_umap <- test_df[, .SD, .SDcols = !c("id")]

umap_model <- umap(train_df_umap)

umap_test_transformed <- predict(umap_model, data = as.matrix(test_df_umap))

train_df_reduced <- as.data.table(umap_model$layout)
test_df_reduced <- as.data.table(umap_test_transformed)

train_df_reduced[is.na(train_df_reduced)] <- 0
test_df_reduced[is.na(test_df_reduced)] <- 0

train_df_final <- cbind(train_df[, ..cols_to_exclude], train_df_reduced)
test_df_final <- cbind(test_df[, c("id")], test_df_reduced)

##########################################################################################
# Prepare data for XGBoost
##########################################################################################
labels <- train_df_final[, .(yZ0, yZ1, yZ2, yZ3, yZ4, yZ5)]
train_labels <- max.col(labels, ties.method = "first") -1 

train_matrix <- xgb.DMatrix(data = as.matrix(train_df_final[, !c("id", "yZ0", "yZ1", "yZ2", "yZ3", "yZ4", "yZ5")]), label = train_labels)
test_matrix <- xgb.DMatrix(data = as.matrix(test_df_final[, !c("id")]))

##########################################################################################
# Train XGBoost Model
##########################################################################################
params <- list(
  objective = "multi:softprob",
  num_class = 6,
  eval_metric = "mlogloss"
)

model <- xgb.train(
  params = params,
  data = train_matrix,
  nrounds = 1000,
  watchlist = list(train = train_matrix),
  verbose = 1,
  early_stopping_rounds = TRUE
)

##########################################################################################
# Make Predictions
##########################################################################################

preds <- predict(model, test_matrix)
preds_matrix <- matrix(preds, ncol = 6, byrow = TRUE)

##########################################################################################
# Prepare Submission File
##########################################################################################

test_ids <- test_df$id
one_hot_preds <- t(apply(preds_matrix, 1, function(row) {
  one_hot <- rep(0, 6)
  one_hot[which.max(row)] <- 1
  return(one_hot)
}))

submission <- data.table(id = test_ids, one_hot_preds)
setnames(submission, c("id", "Z0", "Z1", "Z2", "Z3", "Z4", "Z5"))

write.csv(submission, "submission.csv", row.names = FALSE)



