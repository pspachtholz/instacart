rm(list=ls())
gc()
library(data.table)
library(dplyr)
library(tidyr)
library(lightgbm)
library(stringr)
library(ModelMetrics)


setwd("D:/Eigene Dateien/sonstiges/Kaggle/instacart/scripts")

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
setkey(ord,order_id)
setkey(opp,order_id)

op <- ord[opp,nomatch=0]

# data.table is way faster
setkeyv(op,c("user_id","product_id", "order_number"))
op[,num_order := length(unique(order_id)),.(user_id)]
op[,c("product_time","first_order","second_order", "last_order","sum_order") := .(1:.N,min(order_number),order_number[2], max(order_number),.N),.(user_id,product_id)]

rm(opp)
gc()


# Products ----------------------------------------------------------------

## product_time: how often has a product been ordered total per user
prd2 <- op %>%
  arrange(user_id, order_number, product_id) %>%
  group_by(user_id, product_id) %>%
  mutate(product_time = row_number()) %>%
  ungroup() %>%
  group_by(product_id) %>%
  summarise( #per product
    prod_orders = n(),
    prod_reorders = sum(reordered),
    prod_first_orders = sum(product_time == 1),
    prod_second_orders = sum(product_time == 2)
  )

prd2$prod_reorder_probability <- prd2$prod_second_orders / prd2$prod_first_orders # 
prd$prod_reorder_times <- 1 + prd$prod_reorders / prd$prod_first_orders
prd$prod_reorder_ratio <- prd$prod_reorders / prd$prod_orders

prd <- op[, .(prod_orders = .N, 
              prod_reorders = sum(reordered), 
              prod_first_orders = sum(product_time==1), 
              prod_second_orders = sum(product_time==2), 
              prod_add_to_cart = mean(add_to_cart_order), 
              prod_inpercent_orders=mean(sum_order)/mean(num_order), 
              prod_inpercent_afterfirst = mean(sum_order)/(mean(num_order)-mean(first_order)+1),
              prod_popularity = mean(length(unique(user_id))),
              prod_orders_till_reorder = mean(second_order-first_order,na.rm=T)),by=product_id][, 
                  ':=' (prod_reorder_probability = prod_second_orders / prod_first_orders,
                       prod_reorder_times = 1 + prod_reorders / prod_first_orders,
                       prod_reorder_ratio = prod_reorders / prod_orders)]

prd <- prd %>% select(-prod_reorders, -prod_first_orders, -prod_second_orders)

prd <- products %>% mutate(organic = ifelse(str_detect(str_to_lower(product_name),'organic'),1,0)) %>% select(product_id, organic) %>% right_join(prd, by="product_id")


# mean days till first reorder

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

user_ids <- train$user_id
n_users <- n_distinct(train$user_id)

val_users <- sample_n(data.frame(id=unique(train$user_id)),10000)$id
train_users <- setdiff(user_ids, val_users)

val <- train %>% filter(user_id %in% val_users)
train <- train %>% filter(user_id %in% train_users)

val_userid <- val$user_id
train_userid <- train$user_id

subtrain <- sample_frac(train, 1)
subval <- sample_frac(val, 1)

subval_userid <- subval$user_id
subtrain_userid <- subtrain$user_id

train$eval_set <- NULL
train$user_id <- NULL
train$product_id <- NULL
train$order_id <- NULL
train$reordered[is.na(train$reordered)] <- 0

subtrain$eval_set <- NULL
subtrain$user_id <- NULL
subtrain$product_id <- NULL
subtrain$order_id <- NULL
subtrain$reordered[is.na(subtrain$reordered)] <- 0

subval$eval_set <- NULL
subval$user_id <- NULL
subval$product_id <- NULL
subval$order_id <- NULL
subval$reordered[is.na(subval$reordered)] <- 0


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
  "eval_metric"         = "auc",
  "eta"                 = 0.1,
  "max_depth"           = 6,
  "min_child_weight"    = 10,
  "gamma"               = 0.70,
  "subsample"           = 0.76,
  "colsample_bytree"    = 0.9,
  "alpha"               = 2e-05,
  "lambda"              = 10
)



dtrain <- xgb.DMatrix(as.matrix(subtrain %>% select(-reordered)), label = subtrain$reordered)
dval <- xgb.DMatrix(as.matrix(subval %>% select(-reordered)), label = subval$reordered)
watchlist <- list(train = dtrain, val=dval)

model <- xgb.train(data = dtrain, params = params, nrounds = 100, watchlist=watchlist)

df_subtrain <- data.frame(user_id=subtrain_userid, y=subtrain$reordered, yhat=predict(model,dtrain), pred=predict(model,dtrain)>0.18)
tmp <- df_subtrain %>% group_by(user_id) %>% summarise(f1=f1(y, pred)) 
tmp %>% ungroup() %>% summarize(meanf1 = mean(f1)) %>% .[[1]]

df_subval <- data.frame(user_id=subval_userid, y=subval$reordered, yhat=predict(model,dval))
tmp <- df_subval %>% group_by(user_id) %>% summarise(f1=f1Score(y, yhat, cutoff=0.18)) 
tmp %>% mutate(f1=ifelse(is.na(f1),0,f1)) %>% summarize(meanf1 = mean(f1)) %>% .[[1]]


importance <- xgb.importance(colnames(dtrain), model = model)
xgb.ggplot.importance(importance)

rm(dtrain, importance, subtrain)
gc()



# Apply model -------------------------------------------------------------
dtest <- xgb.DMatrix(as.matrix(test %>% select(-order_id, -product_id)))
test$reordered <- predict(model, dtest)

test$reordered <- (test$reordered > 0.18) * 1

mean(test$reordered)

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
