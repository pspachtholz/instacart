rm(list=ls())
gc()
library(data.table)
library(dplyr)
library(tidyr)
library(lightgbm)
library(stringr)
library(ModelMetrics)
library(ggplot2)


#setwd("D:/Eigene Dateien/sonstiges/Kaggle/instacart/scripts")

f1 <- function (y, pred)
{
  tp <- sum(pred==1 & y == 1)
  fp <- sum(pred==1 & y == 0)
  fn <- sum(pred==0 & y == 1)
  
  precision <- ifelse ((tp==0 & fp==0), 0, tp/(tp+fp))
  recall <- ifelse ((tp==0 & fn==0), 0, tp/(tp+fn))
  
  score <- ifelse ((precision==0 & recall==0), 0, 2*precision*recall/(precision+recall))
  score
  
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
setkey(opp,order_id)
setkey(ord,order_id)
op <- merge(ord,opp,all=FALSE) # inner join filter

rm(opp)
gc()


# Takse subset of Data ----------------------------------------------------
test_users <- unique(ord[eval_set=="test", user_id])
train_users <- unique(ord[eval_set=="train", user_id])

n_train_users <- length(train_users)

n_users <- 30000
sel_train_users <- train_users[1:n_users]

all_users <- c(sel_train_users, test_users)

# do the subsetting
op<-op[user_id %in% all_users]
opt<-opt[user_id %in% all_users]
ord<-ord[user_id %in% all_users]



# data.table is way faster
setkeyv(op,c("user_id","product_id", "order_number"))
op[,num_order := length(unique(order_id)),.(user_id)]
op[,c("product_time","first_order","second_order", "last_order","sum_order") := .(1:.N,min(order_number),order_number[2], max(order_number),.N),.(user_id,product_id)]



# Products ----------------------------------------------------------------

prd <- op[, .(
              prod_orders = .N, 
              prod_reorders = sum(reordered), 
              prod_first_orders = sum(product_time==1), 
              prod_second_orders = sum(product_time==2), 
              prod_add_to_cart = mean(add_to_cart_order), 
              prod_inpercent_orders=mean(sum_order)/mean(num_order), 
              prod_inpercent_afterfirst = mean(sum_order)/(mean(num_order)-mean(first_order)+1),
              prod_popularity = mean(length(unique(user_id))),
              prod_orders_till_reorder = mean(second_order-first_order,na.rm=T),
              prod_days_till_reorder = mean(days_since_prior_order,na.rm=T)),by=product_id][,':=' (
                             prod_reorder_probability = prod_second_orders / prod_first_orders,
                             prod_reorder_times = 1 + prod_reorders / prod_first_orders,
                             prod_reorder_ratio = prod_reorders / prod_orders,
                             prod_reorders = NULL, 
                             prod_first_orders = NULL,
                             prod_second_orders = NULL)]

products <- as.data.table(products)
products <- products[,':=' (prod_organic = ifelse(str_detect(str_to_lower(product_name),'organic'),1,0))][,.(product_id, organic)]
setkey(products,product_id)
setkey(prd, product_id)
prd <- merge(prd, products, all.x=TRUE)

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

# get the train and test orders
us <- ord %>%
  filter(eval_set != "prior") %>%
  select(user_id, order_id, eval_set,
         train_time_since_last_order = days_since_prior_order,
         train_dow = order_dow,
         train_how = order_hour_of_day,
         train_ordernum = order_number)

users <- users %>% inner_join(us)

users <- as.data.table(users)

rm(us)
gc()


# Database ----------------------------------------------------------------

data <- op[, .(
              up_orders = .N, 
              up_first_order = min(order_number), 
              up_last_order = max(order_number), 
              up_avg_cart_position = mean(add_to_cart_order)), 
              .(user_id, product_id)]

rm(op, ord)


#### look at merging first: merge....
setkey(prd,product_id)
setkey(data,product_id)
data <- merge(data,prd,all=FALSE)

setkey(users,user_id)
setkey(data,user_id)
data <- merge(data,users,all=FALSE)


rm(prd, users)
gc()

data[,':=' (up_order_rate = up_orders / user_orders,
            up_orders_since_last_order = user_orders - up_last_order,
            up_inpercent_orders = up_orders / user_orders,
            up_inpercent_afterfirst = (up_orders-1)/ (user_orders-up_first_order))]


setkey(opt, user_id, product_id)
setkey(data, user_id, product_id)
data <- merge(data, opt[,.(user_id, product_id, reordered)], all.x=TRUE)

rm(opt)
gc()



# Train / Test datasets ---------------------------------------------------
train <- as.data.frame(data[data$eval_set == "train",])
train_user_id <- train$user_id
train$eval_set <- NULL
train$user_id <- NULL
train$product_id <- NULL
train$order_id <- NULL
train$reordered[is.na(train$reordered)] <- 0

test <- as.data.frame(data[data$eval_set == "test",])
test$eval_set <- NULL
test$user_id <- NULL
test$reordered <- NULL

rm(data)
gc()


# Model -------------------------------------------------------------------
library(xgboost)

params <- list(
  "objective"           = "reg:logistic",
  "eval_metric"         = "logloss",
  "eta"                 = 0.1,
  "max_depth"           = 6,
  "min_child_weight"    = 10,
  "gamma"               = 0.70,
  "subsample"           = 0.76,
  "colsample_bytree"    = 0.9,
  "alpha"               = 2e-05,
  "lambda"              = 10
)

dtrain <- xgb.DMatrix(as.matrix(train %>% select(-reordered)), label = train$reordered)
model <- xgboost(data = dtrain, params = params, nrounds = 70)

importance <- xgb.importance(colnames(dtrain), model = model)
xgb.ggplot.importance(importance)

df_train <- data.frame(user_id=train_user_id, y=train$reordered, yhat=predict(model,dtrain), pred=predict(model,dtrain)>0.18)
tmp <- df_train %>% group_by(user_id) %>% summarise(f1=f1(y, pred)) 
tmp %>% ungroup() %>% summarize(meanf1 = mean(f1)) %>% .[[1]]

rm(X, importance, subtrain)
gc()


# Apply model -------------------------------------------------------------
X <- xgb.DMatrix(as.matrix(test %>% select(-order_id, -product_id)))
test$reordered <- predict(model, X)

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


