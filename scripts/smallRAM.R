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
  
  precision <- ifelse ((tp==0 & fp==0), 0, tp/(tp+fp)) # no reorders predicted
  recall <- ifelse ((tp==0 & fn==0), 0, tp/(tp+fn)) # no products reordered

  score <- ifelse((all(pred==0) & all(y==0)),1,ifelse((precision==0 & recall==0),0,2*precision*recall/(precision+recall)))
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

# add user_id to train orders
opt$user_id <- ord$user_id[match(opt$order_id, ord$order_id)]
train_info <- opt %>% group_by(user_id) %>% summarize(sum_products = n(), sum_reordered = sum(reordered))


# join products with order info for all prior orders
setkey(opp,order_id)
setkey(ord,order_id)
op <- merge(ord,opp,all=FALSE) # inner join filter

rm(opp)
gc()

# Get the only reorderes ------------------------------
tmp <- op[order_number>1,.(mean_reordered = mean(reordered), n=.N), user_id]
reorder_users <- tmp[mean_reordered==1, user_id]
rm(tmp)
gc()



# Take subset of Data ----------------------------------------------------
test_users <- unique(ord[eval_set=="test", user_id])
train_users <- unique(ord[eval_set=="train" & !user_id %in% reorder_users , user_id])

n_train_users <- length(train_users)

n_users <- 25000
sel_train_users <- train_users[1:n_users]

all_users <- c(sel_train_users, test_users)

# do the subsetting
op<-op[user_id %in% all_users]
opt<-opt[user_id %in% all_users]
ord<-ord[user_id %in% all_users]



# data.table is way faster
setkeyv(op,c("user_id","product_id", "order_number"))
op[,num_order := length(unique(order_id)),.(user_id)]
op[, ':=' ( product_time = 1:.N,
            first_order = min(order_number),
            second_order = order_number[2],
            last_order = max(order_number),
            sum_order = .N), .(user_id,product_id)]



# Products ----------------------------------------------------------------

prd <- op[, .(
              prod_orders = .N, 
              prod_reorders = sum(reordered), 
              prod_first_orders = sum(product_time==1), 
              prod_second_orders = sum(product_time==2), 
              prod_add_to_cart = mean(add_to_cart_order), 
              prod_inpercent_orders=mean(sum_order/num_order), 
              prod_inpercent_afterfirst = mean(sum_order/(num_order-first_order+1)),
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
products[, ':=' (prod_organic = ifelse(str_detect(str_to_lower(product_name),'organic'),1,0))]
products[, ':=' (product_name = NULL)]
setkey(products,product_id)
setkey(prd, product_id)
setkey(op, product_id)
prd <- merge(prd, products[,.(product_id, aisle_id, department_id)], all.x=TRUE)

op <- merge(op, products[,.(product_id, aisle_id, department_id)], all.x=TRUE)

rm(products)
gc()


# Users -------------------------------------------------------------------
users <- ord[eval_set=="prior", .(user_orders=max(order_number),
                         user_period=sum(days_since_prior_order, na.rm = T),
                         user_mean_days_since_prior = mean(days_since_prior_order, na.rm = T)), user_id]

us <- op[,.(
  user_total_products = .N,
  user_reorder_ratio = sum(reordered == 1) / sum(order_number > 1),
  user_distinct_products = uniqueN(product_id),
  user_distinct_aisles = uniqueN(aisle_id),
  user_distinct_depts = uniqueN(department_id)
), user_id]

users <- merge(users, us, all=FALSE)
users[,':=' (user_average_basket = user_total_products / user_orders)]

us <- ord[eval_set != "prior", .(user_id,
                                 order_id,
                                 eval_set,
                                 train_time_since_last_order = days_since_prior_order,
                                 train_dow = order_dow,
                                 train_how = order_hour_of_day,
                                 train_ordernum = order_number)]

setkey(users, user_id)
setkey(us, user_id)
users <- merge(users, us, all=FALSE)

rm(us)
gc()


# Database ----------------------------------------------------------------

data <- op[, .(
  up_orders = .N, 
  up_first_order = min(order_number), 
  up_last_order = max(order_number), 
  up_last_order_dow = order_dow[order_number==max(order_number)],
  up_last_order_hod = order_hour_of_day[order_number==max(order_number)],
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
            up_inpercent_afterfirst = up_orders / (user_orders - up_first_order + 1))]


setkey(opt, user_id, product_id)
setkey(data, user_id, product_id)
data <- merge(data, opt[,.(user_id, product_id, reordered)], all.x=TRUE)

rm(opt)
gc()



# Train / Test datasets ---------------------------------------------------
train <- as.data.frame(data[data$eval_set == "train",])

train_user_id <- train$user_id

val_users <- sample_n(data.frame(id=unique(train_user_id)),5000)$id
train_users <- setdiff(train_user_id, val_users)

val <- train %>% filter(user_id %in% val_users)
train <- train %>% filter(user_id %in% train_users)

val_userid <- val$user_id
train_userid <- train$user_id

train$eval_set <- NULL
train$user_id <- NULL
train$product_id <- NULL
train$order_id <- NULL
train$reordered[is.na(train$reordered)] <- 0

val$eval_set <- NULL
val$user_id <- NULL
val$product_id <- NULL
val$order_id <- NULL
val$reordered[is.na(val$reordered)] <- 0

test <- as.data.frame(data[data$eval_set == "test",])
test_user_id <- test$user_id
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
  "max_depth"           = 5,
  "min_child_weight"    = 10,
  "gamma"               = 0.70,
  "subsample"           = 1,
  "colsample_bytree"    = 0.95,
  "alpha"               = 2e-05,
  "lambda"              = 10
)

threshold = 0.18

dtrain <- xgb.DMatrix(as.matrix(train %>% select(-reordered)), label = train$reordered)
dval <- xgb.DMatrix(as.matrix(val %>% select(-reordered)), label = val$reordered)

model <- xgboost(data = dtrain, params = params, nrounds = 160)

importance <- xgb.importance(colnames(dtrain), model = model)
#xgb.ggplot.importance(importance)

df_train <- data.frame(user_id=train_userid, y=train$reordered, yhat=predict(model,dtrain), pred=predict(model,dtrain)>threshold)
tmp <- df_train %>% group_by(user_id) %>% summarise(f1=f1(y, pred), sum_predicted=sum(pred)) %>% left_join(train_info)
tmp <- tmp %>% mutate(diff_basket_size = sum_products-sum_predicted)
#ggplot(tmp,aes(x=sum_predicted,y=f1))+geom_point()+geom_smooth()
#ggplot(tmp,aes(x=sum_reordered/sum_products,y=f1))+geom_point()+geom_smooth()

tmp %>% ungroup() %>% summarize(meanf1 = mean(f1)) %>% .[[1]]


df_val <- data.frame(user_id=val_userid, y=val$reordered, yhat=predict(model,dval), pred=predict(model,dval)>threshold)
tmp <- df_val %>% group_by(user_id) %>% summarise(f1=f1(y, pred), sum_predicted=sum(pred)) %>% left_join(train_info)
tmp <- tmp %>% mutate(diff_basket_size = sum_products-sum_predicted)
#ggplot(tmp,aes(x=sum_predicted,y=f1))+geom_point()+geom_smooth()
#ggplot(tmp,aes(x=sum_reordered/sum_products,y=f1))+geom_point()+geom_smooth()

tmp %>% ungroup() %>% summarize(meanf1 = mean(f1)) %>% .[[1]]

rm(importance)
gc()


# Apply model -------------------------------------------------------------
dtest <- xgb.DMatrix(as.matrix(test %>% select(-order_id, -product_id)))
test$reordered <- predict(model, dtest)

# Threshold ---------------------------------------------------------------
close_orders <- test %>% group_by(order_id) %>% summarize(m=mean(reordered),mx=max(reordered),s=sum(reordered>threshold)) %>% filter(between(m,0.9*threshold,1.1*threshold) & s <= 5 & mx <= 0.35) %>% select(order_id) %>% .[[1]]
test$reordered <- (test$reordered > threshold) * 1

# all reorderes to 1 -----------------------------------------------------
test$user_id <- test_user_id
test <- as.data.table(test)
test[user_id %in% reorder_users, ':=' (reordered=1)]


submission <- test %>%
  filter(reordered == 1) %>%
  group_by(order_id) %>%
  summarise(
    products = paste(product_id, collapse = " ")
  )

# add None to close orders -----------------------------------------------
new_submission <- submission %>% mutate(products = ifelse(order_id %in% close_orders, str_c(products,'None', collapse = " "),products))

missing <- data.frame(
  order_id = unique(test$order_id[!test$order_id %in% new_submission$order_id]),
  products = "None"
)

new_submission <- new_submission %>% bind_rows(missing) %>% arrange(order_id)
write.csv(new_submission, file = "submit.csv", row.names = F)


