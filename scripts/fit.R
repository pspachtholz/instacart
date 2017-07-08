rm(list=ls())
gc()
library(data.table)
library(dplyr)
library(tidyr)
library(lightgbm)


setwd("D:/Eigene Dateien/sonstiges/Kaggle/instacart/scripts")


print("Mean F1 Score for use with XGBoost")
eval_f1 <- function (yhat, dtrain) {
  require(ModelMetrics)
  y <- getinfo(dtrain, "label")
  dt <- data.table(user_id=valid_users, purch=y, pred=yhat)
  dt <- dt %>% group_by(user_id) %>% mutate(f1_1=f1Score(purch, pred, cutoff=0.1), f1_2=f1Score(purch, pred, cutoff=0.2), f1_3=f1Score(purch, pred, cutoff=0.3)) 
  f1 <- mean(dt$f1_1,na.rm=T)
  f2 <- mean(dt$f1_2,na.rm=T)  
  f3 <- mean(dt$f1_3,na.rm=T)
  return (list(name = "f1", value = f1, higher_better = TRUE))
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

# add user_id to train orders
opt$user_id <- ord$user_id[match(opt$order_id, ord$order_id)]

# join products with order info for all prior orders
op <- ord %>% inner_join(opp, by = "order_id")

rm(opp)
gc()


# Products ----------------------------------------------------------------

## product_time: how often has a product been ordered total per user
prd <- op %>%
  arrange(user_id, order_number, product_id) %>%
  group_by(user_id, product_id) %>%
  mutate(product_time = row_number()) %>% head(100) %>% View()
  ungroup() %>%
  group_by(product_id) %>%
  summarise( #per product
    prod_orders = n(),
    prod_reorders = sum(reordered),
    prod_first_orders = sum(product_time == 1),
    prod_second_orders = sum(product_time == 2)
  )

prd$prod_reorder_probability <- prd$prod_second_orders / prd$prod_first_orders
prd$prod_reorder_times <- 1 + prd$prod_reorders / prd$prod_first_orders
prd$prod_reorder_ratio <- prd$prod_reorders / prd$prod_orders

prd <- prd %>% select(-prod_reorders, -prod_first_orders, -prod_second_orders)

prd <- products %>% mutate(prod_organic = ifelse(str_detect(str_to_lower(product_name),'organic'),1,0)) %>% select(product_id, organic) %>% right_join(prd, by="product_id")

prd <- op %>% 
  group_by(user_id) %>% mutate(num_orders=max(order_number)) %>% 
  ungroup() %>% 
  group_by(user_id, product_id) %>%
  mutate(sum_reorders = sum(reordered)) %>% 
  ungroup() %>% 
  mutate(inpercent_orders = sum_reorders/num_orders) %>% 
  group_by(product_id) %>% 
  summarise(prod_inpercent_orders = mean(inpercent_orders)) %>% 
  right_join(prd, by="product_id")

prd <- op %>% group_by(product_id) %>% summarize(prod_popularity=n_distinct(user_id)) %>% 
  right_join(prd, by="product_id")

prd$prod_order_people_diversity <- prd$prod_orders/prd$prod_popularity

prd <- op %>% 
  arrange(user_id, order_number, product_id) %>%
  group_by(user_id, product_id) %>%
  mutate(product_time = row_number()) %>% mutate(prod_orders_till_reorder = nth(order_number,2)-nth(order_number,1)) %>% summarise(prod_orders_till_reorder=mean(prod_orders_till_reorder)) %>% ungroup() %>% group_by(product_id) %>% summarise(prod_orders_till_reorder=mean(prod_orders_till_reorder,na.rm=T)) %>% right_join(prd, by="product_id")

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
test <- as.data.frame(data[data$eval_set == "test",])

# 131,209 users in total
users_per_fold <- 1000
n_fold <- 5
val_users_random <- sample(unique(train$user_id), size = n_fold*users_per_fold, replace = FALSE)
val_user_groups <- cut(val_users_random,n_fold,labels=FALSE)
val_users <- data.frame(user_id=val_users_random, group=val_user_groups)
train <- train %>% left_join(val_users,by="user_id")
train$reordered[is.na(train$reordered)]<-0

folds <- list()
for (i in 1:n_fold) {
  folds[[i]] <- which(train$group == i)
}

train$group <- NULL


params <- list(booster="gbtree"
               ,objective="binary:logistic"
               ,eval_metric='auc'
               ,eta=0.1
               ,gamma=0.7
               ,max_depth=6
               ,subsample=0.76
               ,colsample_bytree=0.95
               ,base_score=0.2
               ,lambda=10
               ,alpha=2e-05
)

n_rounds <- 2
res<-list()
res$f1 <- matrix(0,n_rounds,n_fold)
res$f2 <- matrix(0,n_rounds,n_fold)
res$f3 <- matrix(0,n_rounds,n_fold)
for (i in 1:length(folds)) {
  cv_train <- train[-folds[[i]],]
  cv_test <- train[folds[[i]],]
  dtrain <- xgb.DMatrix(data.matrix(select(cv_train,-reordered,-user_id,-product_id)),label=cv_train$reordered)
  dtest <- xgb.DMatrix(data.matrix(select(cv_test,-reordered,-user_id,-product_id)),label=cv_test$reordered)  
  watchlist <- list(train = dtrain)
  bst <- xgb.train(params,dtrain,1, watchlist = watchlist)  
  for (j in 2:n_rounds){
    bst <- xgb.train(params,dtrain,1, watchlist=watchlist,xgb_model=bst) # incremental boost
    pred<-predict(bst,dtest)
    y <- getinfo(dtest,'label')
    valid_users <- cv_test$user_id
    dt <- data.table(user_id=valid_users, purch=y, pred=pred)
    dt <- dt %>% group_by(user_id) %>% mutate(f1_1=f1Score(purch, pred, cutoff=0.16), f1_2=f1Score(purch, pred, cutoff=0.18), f1_3=f1Score(purch, pred, cutoff=0.20)) 
    f1 <- mean(dt$f1_1,na.rm=T)
    f2 <- mean(dt$f1_2,na.rm=T)  
    f3 <- mean(dt$f1_3,na.rm=T) 
    res$f1[j,i] <- f1
    res$f2[j,i] <- f2
    res$f3[j,i] <- f3
  }
}
res_mean <- sapply(res,function(x) rowMeans(x)) 
best_iter <- arrayInd(which.max(res_mean),dim(res_mean))[1]

model <- xgb.train(params,dtrain,best_iter)

importance <- xgb.importance(colnames(dtrain), model = model)
xgb.ggplot.importance(importance)

rm(importance)
gc()


# Apply model -------------------------------------------------------------
dtest <- xgb.DMatrix(as.matrix(select(test,-user_id,-order_id,-product_id,-eval_set)))
test$reordered <- predict(model, dtest)

test$reordered <- (test$reordered > 0.18) * 1

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



### try lgbm

params <- list(objective = 'binary',
               metric = 'binary_logloss')

cv_res <- list()
for (i in 1:length(folds)) {
  cv_train <- train[-folds[[i]],]
  cv_test <- train[folds[[i]],]
  dtrain <- lgb.Dataset(data.matrix(select(cv_train,-reordered,-user_id,-product_id)),label=cv_train$reordered, free_raw_data = FALSE)
  valid_users <- cv_test$user_id
  dvalid <- lgb.Dataset.create.valid(dtrain, data.matrix(select(cv_test,-reordered,-user_id,-product_id)),label=cv_test$reordered)
  valids <- list(valid=dvalid)
  for (j in 1:10){
    bst <- lgb.train(params,dtrain,1,init_model=bst)
  }
  cv_res[[i]]<-unlist(bst$record_evals$valid$f1$eval)
}

result<-matrix(unlist(cv_res),50,5)
best_iter <- which.max(rowMeans(result))

