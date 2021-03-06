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

f1_xgb <- function(pred, dtrain) {
  y = getinfo(dtrain, "label")
  dt <- data.table(user_id=val_userid, y=y, pred=pred)
  f1 <- dt[,.(f1 = f1(y,(pred>0.18)*1)),user_id][,.(f1=mean(f1))]
  return (list(metric = "f1", value = f1$f1))
}

# Load Data ---------------------------------------------------------------
path <- "../input"

aisles <- fread(file.path(path, "aisles.csv"))
departments <- fread(file.path(path, "departments.csv"))
opp <- fread(file.path(path, "order_products__prior.csv"))
opt <- fread(file.path(path, "order_products__train.csv"))
ord <- fread(file.path(path, "orders.csv"))
products <- fread(file.path(path, "products.csv"))


# add user_id to train orders
opt$user_id <- ord$user_id[match(opt$order_id, ord$order_id)]
train_info <- opt[,.(sum_products = .N, sum_reordered=sum(reordered)),user_id]


# join products with order info for all prior orders
setkey(opp,order_id)
setkey(ord,order_id)
op <- merge(ord,opp,all=FALSE) # inner join filter

rm(opp)
gc()

# Get the only reorderes ------------------------------
reorder_users <- op[order_number>1,.(mean_reordered = mean(reordered), n=.N), user_id][mean_reordered==1,user_id]
gc()



# Take subset of Data ----------------------------------------------------
test_users <- unique(ord[eval_set=="test", user_id])
train_users <- unique(ord[eval_set=="train" & !user_id %in% reorder_users , user_id])
n_users <- 25000
all_train_users <- train_users[1:n_users]

all_users <- c(all_train_users, test_users)

# do the subsetting
op<-op[user_id %in% all_users]
opt<-opt[user_id %in% all_users]
ord<-ord[user_id %in% all_users]

setkeyv(op,c("user_id","product_id", "order_number"))
op[,num_order := uniqueN(order_id),.(user_id)]
ord[,max_order := uniqueN(order_id),.(user_id)]

#op<-op[order_number >= num_order-3]
#ord<-ord[order_number >= max_order-3]

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
              prod_popularity = mean(uniqueN(user_id)),
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
users <- ord[eval_set=="prior", .(user_orders=.N,
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

us <- op[,.(user_order_products = .N),.(user_id,order_id)][,.(user_order_products_min=min(user_order_products),user_order_products_max=max(user_order_products),user_order_products_sd=sd(user_order_products)), user_id]
users <- merge(users, us, all=FALSE)

us <- op[(num_order-order_number)<=1, .(user_order_products_2 = .N, mean_reordered=mean(reordered)), .(user_id, order_id)][,.(user_order_products_2 = mean(user_order_products_2), user_reorder_rate_2=mean(mean_reordered)), user_id]
users <- merge(users, us, all=FALSE)

users[,':=' (user_average_basket = user_total_products / user_orders)]

us <- ord[eval_set != "prior", .(user_id,
                                 order_id,
                                 eval_set,
                                 train_time_since_last_order = days_since_prior_order,
                                 train_dow = order_dow,
                                 train_how = order_hour_of_day,
                                 train_ordernum = order_number)][,':=' (train_time_0 = (train_time_since_last_order==0)*1)]

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

# Crossvalidation ---------------------------------------------------------
# 131,209 users in total
users_per_fold <- 1000
n_fold <- 5

# create the folds
val_users_random <- sample(unique(train_userid), size = n_fold*users_per_fold, replace = FALSE)
val_user_groups <- cut(val_users_random,n_fold,labels=FALSE)
val_users <- data.frame(user_id=val_users_random, group=val_user_groups)
train <- train %>% left_join(val_users,by="user_id")
train$reordered[is.na(train$reordered)]<-0

folds <- list()
for (i in 1:n_fold) {
  folds[[i]] <- which(train$group == i)
}

train$group <- NULL

params <- list(
  "objective"           = "reg:logistic",
  "eval_metric"         = f1_xgb,
  "eta"                 = 0.1,
  "max_depth"           = 5,
  "min_child_weight"    = 10,
  "gamma"               = 0.70,
  "subsample"           = 1,
  "colsample_bytree"    = 0.7,
  "alpha"               = 2e-05,
  "lambda"              = 10
)

n_rounds <- 2
res<-list()
res$f1 <- matrix(0,n_rounds,n_fold)
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
    dt <- data.table(user_id=valid_users, y=y, pred=pred)
    dt <- dt %>% group_by(user_id) %>% mutate(f1_1=f1(y, (pred>0.18)*1)) 
    f1 <- mean(dt$f1_1,na.rm=T)
    res$f1[j,i] <- f1
  }
}
res_mean <- sapply(res,function(x) rowMeans(x)) 
best_iter <- arrayInd(which.max(res_mean),dim(res_mean))[1]



# Model -------------------------------------------------------------------
library(xgboost)

params <- list(
  "objective"           = "reg:logistic",
  "eval_metric"         = f1_xgb,
  "eta"                 = 0.1,
  "max_depth"           = 5,
  "min_child_weight"    = 10,
  "gamma"               = 0.70,
  "subsample"           = 1,
  "colsample_bytree"    = 0.7,
  "alpha"               = 2e-05,
  "lambda"              = 10
)

threshold = 0.18

dtrain <- xgb.DMatrix(as.matrix(train %>% select(-reordered)), label = train$reordered)
dval <- xgb.DMatrix(as.matrix(val %>% select(-reordered)), label = val$reordered)

watchlist <- list(val = dval)

model <- xgb.train(data = dtrain, params = params, nrounds = 80, watchlist=watchlist)





importance <- xgb.importance(colnames(dtrain), model = model)
xgb.ggplot.importance(importance)

# Threshold ---------------------------------------------------------------
train<-as.data.table(train)
train$user_id <- train_userid
setkey(train, user_id)

train <- merge(train,train_info, all.x=TRUE)

train$prediction <- predict(model,dtrain)
train <- train[order(user_id,-prediction)]
train[,':=' (top = 1:.N), user_id]
train[, ':=' (pred=ifelse(top<=user_order_products_2*user_reorder_rate_2,1,0))]
train[, ':=' (pred=ifelse(prediction>0.18,1,0))] # fixed threshold
train[, ':=' (pred=ifelse(top<=sum_reordered,1,0))]

tmp <- train %>% group_by(user_id) %>% summarise(f1=f1(reordered, pred), sum_predicted=sum(pred)) %>% left_join(train_info)
tmp <- tmp %>% mutate(diff_basket_size = sum_reordered-sum_predicted)
#ggplot(tmp,aes(x=sum_predicted,y=f1))+geom_point()+geom_smooth()
#ggplot(tmp,aes(x=sum_reordered/sum_products,y=f1))+geom_point()+geom_smooth()

tmp %>% ungroup() %>% summarize(meanf1 = mean(f1)) %>% .[[1]]


val$prediction <- predict(model, dval)
val$user_id <- val_userid

# Threshold ---------------------------------------------------------------
val<-as.data.table(val)
val <- val[order(user_id,-prediction)]
val[,':=' (top = 1:.N), user_id]
val[, ':=' (pred=ifelse(top<=user_order_products_2*user_reorder_rate_2,1,0))]
val[, ':=' (pred=ifelse(prediction>0.18,1,0))] # fixed threshold


summary(glm(reordered ~ prediction, data=val))
tmp <- val %>% group_by(user_id) %>% summarise(f1=f1(reordered, pred), sum_predicted=sum(pred), avg_size=mean(user_average_basket)) %>% left_join(train_info)
tmp <- tmp %>% mutate(diff_basket_size = sum_reordered-sum_predicted)
#ggplot(tmp,aes(x=sum_predicted,y=f1))+geom_point()+geom_smooth()
#ggplot(tmp,aes(x=sum_reordered/sum_products,y=f1))+geom_point()+geom_smooth()

tmp %>% ungroup() %>% summarize(meanf1 = mean(f1)) %>% .[[1]]

rm(importance)
gc()


# Apply model -------------------------------------------------------------
dtest <- xgb.DMatrix(as.matrix(test %>% select(-order_id, -product_id)))
test$reordered <- predict(model, dtest)
test$user_id <- test_user_id

# Threshold ---------------------------------------------------------------
test<-as.data.table(test)
test <- test[order(user_id,-reordered)]
test[,':=' (top = 1:.N), user_id]
test[, ':=' (reordered=ifelse(top<=user_average_basket*user_reorder_ratio,1,0))]
#close_orders <- test %>% group_by(order_id) %>% summarize(m=mean(reordered),mx=max(reordered),s=sum(reordered>threshold)) %>% filter(between(m,0.9*threshold,1.1*threshold) & s <= 5 & mx <= 0.35) %>% select(order_id) %>% .[[1]]
test$reordered <- (test$reordered > threshold) * 1

# all reorderes to 1 -----------------------------------------------------
test[user_id %in% reorder_users, ':=' (reordered=1)]


submission <- test %>%
  filter(reordered == 1) %>%
  group_by(order_id) %>%
  summarise(
    products = paste(product_id, collapse = " ")
  )

# add None to close orders -----------------------------------------------
#new_submission <- submission %>% mutate(products = ifelse(order_id %in% close_orders, str_c(products,'None', collapse = " "),products))

missing <- data.frame(
  order_id = unique(test$order_id[!test$order_id %in% submission$order_id]),
  products = "None"
)

submission <- submission %>% bind_rows(missing) %>% arrange(order_id)
write.csv(submission, file = "submit.csv", row.names = F)


