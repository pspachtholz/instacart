## to do
# give more weights to reordered trials
# get f1 eval function right
rm(list=ls())
gc()
library(data.table)
library(dplyr)
library(tidyr)


setwd("D:/Eigene Dateien/sonstiges/Kaggle/instacart/scripts")


print("Mean F1 Score for use with XGBoost")
xgb_eval_f1 <- function (yhat, dtrain) {
  require(ModelMetrics)
  y = getinfo(dtrain, "label")
  id = getinfo(dtrain, "weight")
  dt <- data.table(user_id=id, purch=y, pred=yhat)
  dt <- dt %>% group_by(user_id) %>% mutate(f1score=f1Score(purch, pred, cutoff=0.2)) 
  f1 <- mean(dt$f1score,na.rm=T)
  return (list(metric = "f1", value = f1))
}

# Load Data ---------------------------------------------------------------
path <- "../input"

aisles <- fread(file.path(path, "aisles.csv"))
departments <- fread(file.path(path, "departments.csv"))
opp <- fread(file.path(path, "order_products__prior.csv"))
opt <- fread(file.path(path, "order_products__train.csv"))
ord <- fread(file.path(path, "orders.csv"))
products <- fread(file.path(path, "products.csv"))


# Reshape data ------------------------------------------------------------
aisles$aisle <- as.factor(aisles$aisle)
departments$department <- as.factor(departments$department)
ord$eval_set <- as.factor(ord$eval_set)
products$product_name <- as.factor(products$product_name)

products <- products %>% 
  inner_join(aisles) %>% inner_join(departments) %>% 
  select(-aisle_id, -department_id)
rm(aisles, departments)

opt$user_id <- ord$user_id[match(opt$order_id, ord$order_id)]

# join products with order info
op <- ord %>% inner_join(opp, by = "order_id")

rm(opp)
gc()


# Products ----------------------------------------------------------------
prd <- op %>%
  arrange(user_id, order_number, product_id) %>%
  group_by(user_id, product_id) %>%
  mutate(product_time = row_number()) %>%
  ungroup() %>%
  group_by(product_id) %>%
  summarise(
    prod_orders = n(),
    prod_reorders = sum(reordered),
    prod_first_orders = sum(product_time == 1),
    prod_second_orders = sum(product_time == 2)
  )

prd$prod_reorder_probability <- prd$prod_second_orders / prd$prod_first_orders
prd$prod_reorder_times <- 1 + prd$prod_reorders / prd$prod_first_orders
prd$prod_reorder_ratio <- prd$prod_reorders / prd$prod_orders

prd <- prd %>% select(-prod_reorders, -prod_first_orders, -prod_second_orders)

rm(products)
gc()

# Users -------------------------------------------------------------------
users <- ord %>%
  filter(eval_set == "prior") %>%
  group_by(user_id) %>%
  summarise(
    user_orders = max(order_number),
    user_period = sum(days_since_prior_order, na.rm = T),
    user_mean_days_since_prior = mean(days_since_prior_order, na.rm = T)
  )

us <- op %>%
  group_by(user_id) %>%
  summarise(
    user_total_products = n(),
    user_reorder_ratio = sum(reordered == 1) / sum(order_number > 1),
    user_distinct_products = n_distinct(product_id)
  )

users <- users %>% inner_join(us)
users$user_average_basket <- users$user_total_products / users$user_orders

us <- ord %>%
  filter(eval_set != "prior") %>%
  select(user_id, order_id, eval_set,
         time_since_last_order = days_since_prior_order)

users <- users %>% inner_join(us)

rm(us)
gc()


# Database ----------------------------------------------------------------
data <- op %>%
  group_by(user_id, product_id) %>% 
  summarise(
    up_orders = n(),
    up_first_order = min(order_number),
    up_last_order = max(order_number),
    up_average_cart_position = mean(add_to_cart_order))

rm(op, ord)

data <- data %>% 
  inner_join(prd, by = "product_id") %>%
  inner_join(users, by = "user_id")

data$up_order_rate <- data$up_orders / data$user_orders
data$up_orders_since_last_order <- data$user_orders - data$up_last_order
data$up_order_rate_since_first_order <- data$up_orders / (data$user_orders - data$up_first_order + 1)

data <- data %>% 
  left_join(opt %>% select(user_id, product_id, reordered), 
            by = c("user_id", "product_id"))

rm(opt, prd, users)
gc()




# Train / Test datasets ---------------------------------------------------
train <- as.data.frame(data[data$eval_set == "train",])
train <- train %>% sample_frac(0.2)


# Crossvalidation using xgb.cv
# 131,209 users in total
users_per_fold <- 1000
n_fold <- 5
val_users_random <- sample(unique(train$user_id), size = n_fold*users_per_fold, replace = FALSE)
val_user_groups <- cut(val_users_random,n_fold,labels=FALSE)
val_users <- data.frame(user_id=val_users_random, group=val_user_groups)
train <- train %>% left_join(val_users,by="user_id")

folds <- list()
for (i in 1:n_fold) {
  folds[[i]] <- which(train$group == i)
}
train$group<-NULL

rm(data)
gc()


train_user_id <- train$user_id

varnames <- setdiff(colnames(train), c("user_id","order_id","eval_set", "product_id"))
train <- train[,varnames]

train$reordered[is.na(train$reordered)] <- 0


test <- as.data.frame(data[data$eval_set == "test",])
varnames <- setdiff(colnames(test), c("user_id","eval_set","reordered"))
test <- test[,varnames]



# Model -------------------------------------------------------------------

library(xgboost)
dtrain <- xgb.DMatrix(data=data.matrix(select(train,-reordered)), label=train$reordered)
rm(train)


params <- list(booster="gbtree"
               ,objective="reg:logistic"
               ,eval_metric=xgb_eval_f1
               ,eta=0.1
               ,gamma=0.7
               ,max_depth=6
               ,subsample=0.76
               ,colsample_bytree=0.95
               ,base_score=0.2
               ,lambda=10
               ,nthread=8
               ,alpha=2e-05
)

res <- xgb.cv(data=dtrain, params=params, nrounds=100, folds=folds)
model <- xgb.train(data = dtrain, params = params, watchlist = watchlist, nrounds = 50)

importance <- xgb.importance(colnames(dtrain), model = model)
xgb.ggplot.importance(importance)

rm(X, importance)
gc()


# Apply model -------------------------------------------------------------
dtest <- xgb.DMatrix(as.matrix(select(test,-order_id,-product_id)))
test$reordered <- predict(model, dtest)

test$reordered <- (test$reordered > 0.2) * 1

submission <- test %>%
  filter(reordered == 1) %>%
  group_by(order_id) %>%
  summarise(
    products = paste(product_id, collapse = " ")
  )

missing <- data.frame(
  order_id = unique(test$order_id[!test$order_id %in% submission$order_id]),
  products = "None"
)

submission <- submission %>% bind_rows(missing) %>% arrange(order_id)
write.csv(submission, file = "submit.csv", row.names = F)

