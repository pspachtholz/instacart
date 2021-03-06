rm(list=ls())
gc()
library(data.table)
library(dplyr)
library(tidyr)
library(xgboost)
library(stringr)
library(ModelMetrics)
library(ggplot2)
library(lightgbm)

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
  dt <- data.table(user_id=train_users, y=y, pred=pred)
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
train_users <- unique(ord[eval_set=="train", user_id]) #& !user_id %in% reorder_users
n_users <- 30000
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
              prod_maxorders = max(product_time),
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

# typical time till reorder
tmp <- op[,.(cumsum=cumsum(days_since_prior_order), reordered), .(product_id, user_id, order_id)][reordered==1][,cumlag:=lag(cumsum),user_id]
pd <- tmp[is.na(cumlag), cumlag:=0][,.(prod_typical_reorderdays = mean(cumsum-cumlag)), product_id]

setkey(pd,product_id)
prd <- merge(prd, pd, all.x=TRUE)


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
), user_id][,':=' (user_pct_distinct_products = user_distinct_products / user_total_products,
                   user_pct_distinct_aisles = user_distinct_aisles / user_total_products,
                   user_pct_distinct_depts = user_distinct_depts / user_total_products)]



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

# Departments -------------------------------------------------------------

dep <- op[,.(dept_total_products = .N,
             dept_total_orders = uniqueN(order_id),
             dept_total_users = uniqueN(user_id),
             dept_reorder_times = sum(reordered),
             dept_reorder_ratio = mean(reordered)), department_id][,':=' 
                      (dept_products_per_order = dept_total_products / dept_total_orders)]

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


setkey(prd,product_id)
setkey(data,product_id)
data <- merge(data,prd,all=FALSE)

setkey(users,user_id)
setkey(data,user_id)
data <- merge(data,users,all=FALSE)

setkey(dep,department_id)
setkey(data,department_id)
data <- merge(data,dep,all=FALSE)


rm(prd, users, dep)
gc()

data[,':=' (up_order_rate = up_orders / user_orders,
            up_orders_since_last_order = user_orders - up_last_order,
            up_inpercent_afterfirst = up_orders / (user_orders - up_first_order + 1),
            up_delta_time_typicaltime = abs(train_time_since_last_order - prod_typical_reorderdays))]


# merge in train order
setkey(opt, user_id, product_id)
setkey(data, user_id, product_id)
data <- merge(data, opt[,.(user_id, product_id, reordered)], all.x=TRUE)

rm(opt)
gc()



# Train / Test datasets ---------------------------------------------------
train <- as.data.frame(data[data$eval_set == "train",])

train_userid <- train$user_id

train$eval_set <- NULL
train$product_id <- NULL
train$order_id <- NULL
train$reordered[is.na(train$reordered)] <- 0

test <- as.data.frame(data[data$eval_set == "test",])
test_user_id <- test$user_id
test$eval_set <- NULL
test$user_id <- NULL
test$reordered <- NULL

rm(data)
gc()

# Model fitting ---------------------------------------------------------
colnames(train)

# Setting params for fitting
params <- list(
  objective           = "binary",
  metric              = "binary_logloss",
  num_leaves          = 96,
  max_depth           = 10,
  feature_fraction    = 0.9,
  bagging_fraction    = 0.95,
  bagging_freq        = 5
)

# Get the folds ---------------------------------

# 131,209 users in total
users_per_fold <- 10000
n_fold <- 3

# create the folds
val_users_random <- sample(unique(train_userid), size = n_fold*users_per_fold, replace = FALSE)
if (n_fold ==1) {
  val_user_groups <- 1
} else {
  val_user_groups <- cut(1:length(val_users_random),n_fold,labels=FALSE)
}
folds <- list()
for (i in 1:n_fold) {
  folds[[i]] <- which(train_userid %in% val_users_random[val_user_groups==i])
}

# Do the CV ------------------------------------

n_rounds <- 100
calc_f1_every_n <- 10
res<-list()
res$f1 <- matrix(0,n_rounds/calc_f1_every_n,n_fold)
for (i in 1:length(folds)) {
  cat('Training on fold', i,'...\n')
  cv_train <- train[-folds[[i]],]
  cv_val <- train[folds[[i]],]
  xtrain <- data.matrix(select(cv_train,-user_id,-reordered))
  ytrain <- cv_train$reordered
  dtrain <- lgb.Dataset(xtrain,label=ytrain)
  xval <- data.matrix(select(cv_val,-user_id,-reordered))
  yval <- cv_val$reordered
  dval <- lgb.Dataset(xval,label=yval)  
  valids <- list(valid = dtrain)
  
  train_users <- cv_train$user_id
  
  for (j in 1:(n_rounds/calc_f1_every_n)){
    bst <- lgb.train(params,dtrain,calc_f1_every_n, valids=valids) # first boosting iteration

    ptrain <- predict(bst, xtrain, rawscore = TRUE)
    setinfo(dtrain, "init_score", ptrain)
    
    pred<-predict(bst,xval)
    y <- yval
    valid_users <- cv_val$user_id
  
    dt <- data.table(user_id=valid_users, y=y, pred=pred)
    f1_score <- dt[,.(f1score = f1(y,(pred>0.18)*1)), user_id][,.(f1_mean=mean(f1score))]
    cat('val-f1: ', f1_score$f1_mean)
    res$f1[j,i] <- f1_score$f1_mean
  }
}
results <- data.frame(m=rowMeans(res$f1),sd=apply(res$f1,1,sd))
results
best_iter <- which.max(results$m)*calc_f1_every_n

n_rounds <- best_iter


# Fit the Model to all training data -------------------------------------
threshold = 0.18

dtrain <- xgb.DMatrix(as.matrix(train %>% select(-user_id,-reordered)), label = train$reordered)

watchlist <- list(train = dtrain)

model <- xgb.train(data = dtrain, params = params, nrounds = n_rounds, watchlist=watchlist)

importance <- xgb.importance(colnames(dtrain), model = model)
xgb.ggplot.importance(importance)+theme(axis.text.y = element_text(hjust = 0))


# Look at predictions ---------------------------------------------------------------
train<-as.data.table(train)
setkey(train, user_id)

train <- merge(train,train_info, all.x=TRUE)

train$prediction <- predict(model,dtrain)
train <- train[order(user_id,-prediction)]
train[,':=' (top = 1:.N), user_id]

ttmp <- train[,.(sp=sum(prediction),sr=sum(reordered)),user_id]
ggplot(ttmp,aes(sp,sr))+geom_point()+geom_abline(slope=1)
cor(ttmp$sp,ttmp$sr)

# choose a threshold -----------------------------------------------------------

train[, ':=' (pred=ifelse(top<=user_order_products_2*user_reorder_rate_2,1,0))]
train[, ':=' (pred=ifelse(prediction>0.18,1,0))] # fixed threshold
train[, ':=' (pred=ifelse(top<=sum_reordered,1,0))]

summary(glm(reordered ~ prediction, data=train))
tmp <- train %>% group_by(user_id) %>% summarise(f1=f1(reordered, pred), sum_predicted=sum(pred)) %>% left_join(train_info)
tmp <- tmp %>% mutate(diff_basket_size = sum_reordered-sum_predicted)
#ggplot(tmp,aes(x=sum_predicted,y=f1))+geom_point()+geom_smooth()
#ggplot(tmp,aes(x=sum_reordered/sum_products,y=f1))+geom_point()+geom_smooth()

tmp %>% ungroup() %>% summarize(meanf1 = mean(f1)) %>% .[[1]]


cv_val$prediction <- predict(model, dval)

# Threshold ---------------------------------------------------------------
cv_val <- cv_val[order(user_id,-prediction)]
cv_val[,':=' (top = 1:.N), user_id]

ttmp <- cv_val[,.(sp=sum(prediction),sr=sum(reordered)),user_id]
ggplot(ttmp,aes(sp,sr))+geom_point()+geom_abline(slope=1)+geom_smooth(method="lm")
cor(ttmp$sp,ttmp$sr)



summary(glm(reordered ~ prediction, data=val))
tmp <- val %>% group_by(user_id) %>% summarise(f1=f1(reordered, pred), sum_predicted=sum(pred), avg_size=mean(user_average_basket)) %>% left_join(train_info)
tmp <- tmp %>% mutate(diff_basket_size = sum_reordered-sum_predicted)
#ggplot(tmp,aes(x=sum_predicted,y=f1))+geom_point()+geom_smooth()
#ggplot(tmp,aes(x=sum_reordered/sum_products,y=f1))+geom_point()+geom_smooth()

tmp %>% ungroup() %>% summarize(meanf1 = mean(f1)) %>% .[[1]]

rm(importance)
gc()


# Apply model to test data ------------------------------------------------
dtest <- xgb.DMatrix(as.matrix(test %>% select(-order_id, -product_id)))
test$pred <- predict(model, dtest)
test$user_id <- test_user_id

# Threshold ---------------------------------------------------------------
test<-as.data.table(test)
test <- test[order(user_id,-pred)]
test[,':=' (top = 1:.N), user_id]
test[,':=' (pred_basket = sum(pred)), user_id]

test[pred_basket>30,':=' (adapted_basket = pred_basket+5)]
test[pred_basket<=30,':=' (adapted_basket = pred_basket)]
test[, ':=' (reordered=ifelse(top<=adapted_basket,1,0))]
#test[, ':=' (reordered=ifelse(top<=user_average_basket*user_reorder_ratio,1,0))]
#close_orders <- test %>% group_by(order_id) %>% summarize(m=mean(reordered),mx=max(reordered),s=sum(reordered>threshold)) %>% filter(between(m,0.9*threshold,1.1*threshold) & s <= 5 & mx <= 0.35) %>% select(order_id) %>% .[[1]]
#test$reordered <- (test$pred > threshold) * 1

# all reorderes to 1 -----------------------------------------------------
#test[user_id %in% reorder_users, ':=' (reordered=1)]

submission <- test[reordered==1,.(products = paste(product_id, collapse = " ")), order_id]

# add None to close orders -----------------------------------------------
#new_submission <- submission %>% mutate(products = ifelse(order_id %in% close_orders, str_c(products,'None', collapse = " "),products))

missing <- data.frame(
  order_id = unique(test$order_id[!test$order_id %in% submission$order_id]),
  products = "None"
)

submission <- submission %>% bind_rows(missing) %>% arrange(order_id)
fwrite(submission, file = "submit.csv")


