rm(list=ls())
gc()
library(data.table)
library(dplyr)
library(lightgbm)
library(stringr)
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

f1m <- function(y,pred){
  y_products <- unlist(str_split(y," "))
  pred_products <- unlist(str_split(pred, " "))
  
  tp <- sum(pred_products %in% y_products)
  fp <- sum(!pred_products %in% y_products)
  fn <- sum(!y_products %in% pred_products)
  
  precision <- tp/(tp+fp)
  recall <- tp/(tp+fn)
  
  score <- ifelse(precision+recall==0,0,2*precision*recall / (precision+recall))
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
reorder_users <- op[order_number>1 & .N>2,.(mean_reordered = mean(reordered), n=.N), user_id][mean_reordered==1,user_id]
gc()


# Take subset of Data ----------------------------------------------------
test_users <- unique(ord[eval_set=="test", user_id])
train_users <- unique(ord[eval_set=="train", user_id]) #& !user_id %in% reorder_users
n_users <- 15000
all_train_users <- train_users[1:n_users]

all_users <- c(all_train_users, test_users)



setkeyv(op,c("user_id","product_id", "order_number"))
op[,last_prior_order := max(order_number),.(user_id)]
ord[,last_order := max(order_number),.(user_id)] # auch train/test orders mit drin

#op<-op[last_prior_order-order_number <= 2] #last order = last prior order
#ord<-ord[last_order-order_number <= 3]

setkey(ord, user_id, order_number)
ord[order(order_number), ':=' (order_days_sum = cumsum(ifelse(is.na(days_since_prior_order),0,days_since_prior_order))),user_id][,':=' (order_days_max=max(order_days_sum)),user_id][, ':=' (order_day_year = 365-(order_days_max-order_days_sum)),user_id]

op <- merge(op, ord[,.(user_id, order_number, order_days_sum, order_days_max, order_day_year)], all.x=T)

op[order(user_id, product_id, order_number), ':=' ( 
            product_time = 1:.N,
            first_order = min(order_number),
            second_order = order_number[2],
            third_order = order_number[3],
            up_sum_order = .N), .(user_id,product_id)]

op[(reordered==1 | product_time==1),':=' (order_days_lag=c(NA,order_days_sum[-.N])), .(user_id, product_id)]


# Products ----------------------------------------------------------------

prd <- op[, .(
              prod_orders = .N, 
              prod_reorders = sum(reordered), 
              prod_first_orders = sum(product_time==1), 
              prod_second_orders = sum(product_time==2), 
              prod_add_to_cart = mean(add_to_cart_order), 
              prod_inpercent_orders=mean(up_sum_order/last_prior_order), 
              prod_inpercent_afterfirst = mean(up_sum_order/(last_prior_order-first_order+1)),
              prod_popularity = mean(uniqueN(user_id)),
              prod_season = mean(order_day_year), 
              prod_orders_till_first_reorder = mean(second_order-first_order,na.rm=T)
              ), product_id][,':=' (
                             prod_reorder_probability = prod_second_orders / prod_first_orders,
                             prod_reorder_times = 1 + prod_reorders / prod_first_orders,
                             prod_reorder_ratio = prod_reorders / prod_orders,
                             prod_reorders = NULL, 
                             prod_first_orders = NULL,
                             prod_second_orders = NULL
                             )]

# do the subsetting after product features were created
op<-op[user_id %in% all_users]
opt<-opt[user_id %in% all_users]
ord<-ord[user_id %in% all_users]

#products[, ':=' (prod_organic = ifelse(str_detect(str_to_lower(product_name),'organic'),1,0))]
#products[, ':=' (product_name = NULL)]
setkey(products,product_id)
setkey(prd, product_id)
setkey(op, product_id)
#prd <- merge(prd, products[,.(product_id, aisle_id, department_id)], all.x=TRUE)

op <- merge(op, products[,.(product_id, aisle_id, department_id)], all.x=TRUE)

rm(products)
gc()

# Order typicality --------------------------------------------------------
od <- ord[order(user_id, order_number), .(order_dow_typicality = (order_dow-order_dow[.N])*1), user_id][
        order_dow_typicality<= -4, ':=' (order_dow_typicality=order_dow_typicality+7)][
          order_dow_typicality >= 4, ':=' (order_dow_typicality=order_dow_typicality-7)
        ]
tmp <- ord[order(user_id, order_number), .(
  order_dow_typicality2 = (sum(order_dow==order_dow[.N])-1)/.N)
  , .(user_id)]

od <- merge(od,tmp)

od <- od[, .(order_dow_typ = mean(abs(order_dow_typicality)), order_dow_typ2 = mean(order_dow_typicality2)), user_id]

tmp <- ord[order(user_id,order_number), .(order_number,
  order_hod_typicality = order_hour_of_day[1:.N]-order_hour_of_day[.N])
, .(user_id)]

tmp[order_hod_typicality < -12,':=' (order_hod_typicality2=order_hod_typicality+24)]
tmp[order_hod_typicality > 12,':=' (order_hod_typicality2=order_hod_typicality-24)]
tmp[between(order_hod_typicality,-12,12),':=' (order_hod_typicality2=order_hod_typicality*1)]
tmp[,':=' (order_hod_typicality = order_hod_typicality2)]
tmp[,':=' (order_hod_typicality2 = NULL)]
tmp <- tmp[, .(order_hod_typicality = mean(order_hod_typicality)), user_id]

od <- merge(od, tmp, by="user_id")

# Users -------------------------------------------------------------------
users <- ord[eval_set=="prior", .(user_orders=.N,
                         user_period=sum(days_since_prior_order, na.rm = T),
                         user_mean_days_since_prior = mean(days_since_prior_order, na.rm = T),
                         user_std_days_since_prior_order = sd(days_since_prior_order, na.rm=T)
                         ), user_id]

tmp <- op[,.(order_sum_products = .N, order_number),.(user_id, order_id)]
tmp[, ':=' (user_mean_basket = mean(order_sum_products), user_mean_order_number = mean(order_number)), user_id]
tmpp <- tmp[, .(order_sum_products = mean(order_sum_products), user_mean_basket = mean(user_mean_basket), user_mean_order_number = mean(user_mean_order_number)), .(user_id, order_number)]
us <- tmpp[, .(user_slope_basket = sum((order_sum_products-user_mean_basket)*(order_number-user_mean_order_number)) / sum((order_number-user_mean_order_number)^2)), user_id]

users <- merge(users,us)
rm(tmp, tmpp)
gc()

tmp <- op[order_number>1, .(pct_reordered = mean(reordered)), .(user_id, order_number)]
us <- tmp[,.(user_reorder_slope = sum((pct_reordered-mean(pct_reordered))*(order_number-1 - mean(order_number-1)))/sum((order_number-1 - mean(order_number-1))^2)), user_id]
users <- merge(users,us)


us <- op[,.(
  user_total_products = .N,
  user_reorder_ratio = sum(reordered == 1) / sum(order_number > 1),
  user_distinct_products = uniqueN(product_id),
  user_distinct_aisles = uniqueN(aisle_id),
  user_distinct_depts = uniqueN(department_id)
), user_id][,':=' (user_pct_distinct_products = user_distinct_products / user_total_products,
                   user_pct_distinct_aisles = user_distinct_aisles / user_total_products,
                   user_pct_distinct_depts = user_distinct_depts / user_total_products,
                   user_distinct_products = NULL,
                   user_distinct_aisles = NULL,
                   user_distinct_depts = NULL)]

users <- merge(users, us, all=FALSE)

us <- op[,.(user_order_products = .N),.(user_id,order_id)][,.(
        user_order_products_mean=mean(user_order_products),
        user_order_products_sd=sd(user_order_products)
        ), user_id]

users <- merge(users, us, all=FALSE)

us <- op[(last_prior_order-order_number)<=2, .(
  user_order_products_3 = .N, 
  user_reorder_ratio_3=mean(reordered)
  ), .(user_id)][,.(
      user_order_products_mean_last3 = mean(user_order_products_3), 
      user_reorder_ratio_last3=mean(user_reorder_ratio_3)
      ), user_id]

users <- merge(users, us, all=FALSE)

users[, ':=' (
  user_recent_orders_factor = user_order_products_mean_last3/user_order_products_mean,
  user_recent_reorder_factor = user_reorder_ratio_last3 / user_reorder_ratio,
  user_activity_products = ifelse(user_period==0,0,user_total_products/user_period),
  user_activity_orders = ifelse(user_period==0,0,user_orders/user_period),
  user_order_products_mean_last3 = NULL,
  user_reorder_ratio_last3 = NULL
  )]

us <- ord[eval_set != "prior", .(
      user_id,
      order_id,
      eval_set,
      train_days_since_last_order = days_since_prior_order,
      train_30days = (days_since_prior_order == 30)*1,
      train_dow = order_dow,
      train_hod = order_hour_of_day)]

setkey(users, user_id)
setkey(us, user_id)
users <- merge(users, us, all=FALSE)

users <- merge(users, od, all=FALSE)

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
# to do days until the product is first reordered
data <- op[, .(
  up_orders = .N, 
  up_first_order = min(order_number), 
  up_last_order = max(order_number), 
  up_last_order_dow = order_dow[order_number==max(order_number)],
  up_last_order_hod = order_hour_of_day[order_number==max(order_number)],
  up_avg_cart_position = mean(add_to_cart_order),
  up_avg_days_since_reorder = mean(order_days_sum-order_days_lag,na.rm=T)),
  .(user_id, product_id)]

tmp <- op[, .(up_days_till_first_reorder=ifelse(is.null(order_days_sum[product_time==2]),NA,order_days_sum[product_time==2]-order_days_sum[product_time==1])), .(user_id, product_id)]
data <- merge(data,tmp, all.x=TRUE)

rm(tmp)
gc()

setkey(users,user_id)
setkey(data,user_id)
data <- merge(data,users,all=FALSE)

data<-merge(data, ord[,.(user_id, order_number, order_days_sum)], all.x=TRUE, by.x=c("user_id", "up_last_order"), by.y=c("user_id", "order_number"))
data[,':=' (
  up_days_since_last_reorder = user_period-order_days_sum+train_days_since_last_order,
  order_days_sum=NULL
)]

setkey(prd,product_id)
setkey(data,product_id)
data <- merge(data,prd,all=FALSE)

rm(op, ord)

data[,':=' (
  up_diff_train_typical = abs(train_days_since_last_order-up_avg_days_since_reorder),
  up_perc_diff_train_typical = train_days_since_last_order/up_avg_days_since_reorder
  )]

#setkey(dep,department_id)
#setkey(data,department_id)
#data <- merge(data,dep,all=FALSE)

rm(prd, users, dep)
gc()

data[,':=' (up_order_rate = up_orders / user_orders,
            up_orders_since_last_order = user_orders - up_last_order,
            up_inpercent_afterfirst = up_orders / (user_orders - up_first_order + 1)
            )]


# merge in train order
setkey(opt, user_id, product_id)
setkey(data, user_id, product_id)
data <- merge(data, opt[,.(user_id, product_id, reordered)], all.x=TRUE)

rm(opt)
gc()


# do it the data.table way

# Train / Test datasets ---------------------------------------------------
train <- data[eval_set == "train"]

train[,':=' (eval_set=NULL)]
train[is.na(reordered), ':=' (reordered=0)]


test <-data[eval_set == "test"]
test[,':=' (eval_set=NULL, reordered=NULL)]


rm(data)
gc()

# Model fitting ---------------------------------------------------------
names(train)

# Setting params for fitting
params <- list(
  objective           = "binary",
  metric              = "binary_logloss",
  num_leaves          = 63,
  max_depth           = 6,
  learning_rate       = 0.03,
  feature_fraction    = 0.95,
  bagging_fraction    = 0.9,
  bagging_freq        = 4,
  min_sum_hessian_in_leaf = 0.1  
)


# Get the folds ---------------------------------

# 131,209 users in total
users_per_fold <- 5000
n_fold <- 3

# create the folds
val_users_random <- sample(unique(train[,user_id]), size = n_fold*users_per_fold, replace = FALSE)
if (n_fold ==1) {
  val_user_groups <- 1
} else {
  val_user_groups <- cut(1:length(val_users_random),n_fold,labels=FALSE)
}
folds <- list()
for (i in 1:n_fold) {
  folds[[i]] <- which(train[,user_id] %in% val_users_random[val_user_groups==i])
}

# Do the CV ------------------------------------
threshold <- 0.20
n_rounds <- 200

calc_f1_rounds <- seq(100,200,10)
bst_rnds <- diff(c(0,calc_f1_rounds, n_rounds))
bst_rnds <- bst_rnds[!bst_rnds==0]

res<-list()
res$f1 <- matrix(0,length(bst_rnds),n_fold)
res$mean_reordered <- matrix(0,length(bst_rnds),n_fold)
for (i in 1:length(folds)) {
  cat('\n\nTraining on fold', i,'...\n')
  cv_train <- train[-folds[[i]],]
  cv_val <- train[folds[[i]],]
  
  xtrain <- as.matrix(cv_train[,-c("user_id", "product_id", "order_id", "reordered"),with=FALSE])
  ytrain <- cv_train[,reordered]
  dtrain <- lgb.Dataset(xtrain,label=ytrain, free_raw_data = FALSE)
  
  xval <- data.matrix(cv_val[,-c("user_id", "product_id", "order_id", "reordered"), with=FALSE])
  yval <- cv_val[,reordered]
  dval <- lgb.Dataset(xval,label=yval, free_raw_data = FALSE)
  
  valids <- list(train = dtrain, valid=dval)  
  
  train_users <- cv_train[,user_id]
  
  for (j in 1:length(bst_rnds)){
    cat('\n','round: ', j, ' total boosting rounds: ', sum(bst_rnds[1:j]), ' n_bst_rounds: ', bst_rnds[j])
    if (j==1){
      bst <- lgb.train(params,dtrain,bst_rnds[j], valids=valids, verbose=0) # first boosting iteration
    } else {
      bst <- lgb.train(params,dtrain,bst_rnds[j], valids=valids, init_model = bst, verbose = 0)
    }
    
    pred<-predict(bst,xval)
    y <- yval
    valid_users <- cv_val[,user_id]
    valid_orders <- cv_val[,order_id]
  
    dt <- data.table(user_id=valid_users, order_id=valid_orders, y=y, pred=pred, ypred=(pred>threshold)*1)
    f1_score <- dt[,.(f1score = f1(y,ypred)), user_id][,.(f1_mean=mean(f1score))]
    cat('val-f1: ', f1_score$f1_mean, 'mean sum_pred: ',dt[,.(sp = sum(ypred)),user_id][,.(mean_sp = mean(sp))]$mean_sp, '\n')
    res$f1[j,i] <- f1_score$f1_mean
    res$mean_reordered[j,i] <- mean(cv_val$reordered)     
  }
  rm(dtrain, dval, bst,cv_train, cv_val, dt, xtrain, xval, ytrain, yval)
  gc()
}
results <- data.frame(m=rowMeans(res$f1),sd=apply(res$f1,1,sd),res$f1, res$mean_reordered)
results
best_iter <- sum(bst_rnds[1:(which.max(as.matrix(results$m))+1)])

n_rounds <- best_iter


# Fit the User Product Model to all training data & predict test ---------------------------------
xtrain <- as.matrix(train[,-c("user_id", "product_id", "order_id", "reordered"),with=FALSE])
ytrain <- train$reordered
dtrain <- lgb.Dataset(xtrain,label=ytrain)

valids <- list(train = dtrain)

model <- lgb.train(data = dtrain, params = params, nrounds = n_rounds, valids=valids)

xtest <- as.matrix(test[,-c("user_id","order_id", "product_id"),with=FALSE])
test$pred <- predict(model, xtest)

best_iter <- 130

# Get oof predictions for best round ----------------------------------------
train$oof_pred <- NA
for (i in 1:length(folds)) {
  cat('\n\nTraining on fold', i,'...\n')
  cv_train <- train[-folds[[i]],]
  cv_val <- train[folds[[i]],]
  
  xtrain <- as.matrix(cv_train[,-c("user_id", "product_id", "order_id", "reordered", "oof_pred"),with=FALSE])
  ytrain <- cv_train$reordered
  dtrain <- lgb.Dataset(xtrain,label=ytrain,free_raw_data = FALSE)
  
  xval <- data.matrix(cv_val[,-c("user_id", "product_id", "order_id", "reordered"), with=FALSE])
  yval <- cv_val$reordered
  dval <- lgb.Dataset(xval,label=yval,free_raw_data = FALSE)
  
  train_users <- cv_train$user_id
  valid_users <- cv_val$user_id
  valid_orders <- cv_val$order_id  
  
  bst_rnds <- best_iter
  bst <- lgb.train(params,dtrain,bst_rnds,verbose=0) # first boosting iteration

  pred<-predict(bst,xval)
  train$oof_pred[folds[[i]]] <- pred

  rm(dtrain, dval, bst, cv_train, cv_val)
  gc()
}

submission1 <- train[(oof_pred>0.19)*1==1,.(ypred = paste(product_id, collapse = " ")), order_id]
submission2 <- train[reordered==1,.(y = paste(product_id, collapse = " ")), order_id]

missing1 <- data.table(
  order_id = unique(train$order_id[!train$order_id %in% submission1$order_id]),
  ypred = "None"
)
missing2 <- data.table(
  order_id = unique(train$order_id[!train$order_id %in% submission2$order_id]),
  y = "None"
)

submission1 <- rbindlist(list(submission1, missing1))
submission2 <- rbindlist(list(submission2, missing2))

submission <- merge(submission1, submission2, by="order_id")

f1score<-submission[,.(f1score = f1m(y,ypred)), order_id][,.(f1score=mean(f1score))]
f1score
## i was here 

f1q <- function(y,pred,addnone){
  if(addnone){
    pred <- c(pred,1)
    if (all(y==0)) {
      y <- c(y,1)
    } else {
      y <- c(y,0)
    }
  }
  tp <- sum(pred==1 & y == 1)
  fp <- sum(pred==1 & y == 0)
  fn <- sum(pred==0 & y == 1)
  
  precision <- ifelse ((tp==0 & fp==0), 0, tp/(tp+fp)) # no reorders predicted
  recall <- ifelse ((tp==0 & fn==0), 0, tp/(tp+fn)) # no products reordered
  
  score <- ifelse((precision==0 & recall==0),0,2*precision*recall/(precision+recall))
  score
}

f1e <- function(y, probs,k) {
  gt <- do.call("CJ",rep(list(c(0,1)),k))
  p <- probs[1:k]
  
  tmp <- gt*2-1
  tmp <- abs(-sweep(gt,2,p, FUN="+")+1)
  
  ysel <- rep(1,k)
  tmp <- apply(tmp,1,prod)
  res<-matrix(0,2,2^k)
  res[1,]<-apply(gt,1,FUN = function(x) f1q(x,ysel,1))
  res[2,]<-apply(gt,1,FUN = function(x) f1q(x,ysel,0))  
  res <- res %*% tmp
  dt <- data.table(k=k,none=which.max(res),f1=res[which.max(res)])
}

train<-train[order(user_id, -oof_pred)]
tmp <- train[user_id==712, .(y=reordered,probs=oof_pred)]

res <- data.table(k=integer(),none=integer(), f1=double())
for (i in 1:nrow(tmp)) {
  res <- rbindlist(list(res,f1e(tmp$y,tmp$probs,i)))
}


th <- train[,.(
  user_id=user_id,
  order_id = order_id, 
  y=reordered, 
  pred=oof_pred)][,':=' (
    pred_basket = sum(pred),
    round_basket = round(sum(pred)),
    y_basket = sum(y)), order_id][
      order(user_id,-pred)]

f1_score <- th[,.(f1score = f1(y,(pred>0.2)*1)), user_id][,.(f1_mean=mean(f1score))]
f1_score

# th[,':=' (r_basket = round(pred_basket))]
# thresh[,':=' (thresh = 0.2161+basket*0.003159)]
# t2 <- merge(th,thresh, by.x="r_basket", by.y="basket")
# f1_score <- t2[,.(f1score = f1(y,(pred>thresh)*1)), user_id][,.(f1_mean=mean(f1score))]
# f1_score
# 
# threshs <- seq(0.15,0.35, 0.01)
# baskets <- seq(0,max(th$r_basket))
# f1_table <- matrix(NA,length(threshs), length(baskets))
# for (i in 1:length(threshs)) {
#   for (j in 1:length(baskets)) {
#     f1_table[i,j] <- th[r_basket==baskets[j],.(f1score = f1(y,(pred>threshs[i])*1)), user_id][,.(f1_mean=mean(f1score))]$f1_mean
#   }
# }
# 
# baskets <- data.table(baskets,th=threshs[unlist(apply(f1_table, 2, FUN = function(x) ifelse(!all(is.na(x)), which.max(x),4)))])
# t2 <- merge(th,baskets, by.x="pred_basket", by.y="baskets",all.x=TRUE)
# t2[is.na(th), th:=0.19]
# f1_score <- t2[,.(f1score = f1(y,(pred>th)*1)), user_id][,.(f1_mean=mean(f1score))]
# f1_score


### try to predict best threshold per order
# find best threshold

maxsize <- th[,.N, user_id][,max(N)]
mat <- matrix(0,maxsize, maxsize)
for (i in 1:dim(mat)[2]){
  mat[(1:i),i]<-1
}

# predictions are ordered descending, take the top k products
order_ids <- unique(th$order_id)
baskets <- unique(th$round_basket)
best_th <- vector(length=length(baskets))
for (j in 1:length(baskets)) {
  cat(j, 'of', length(baskets),'basket',baskets[j], ' ')
  tmp <- th[round_basket==baskets[j]]
  user_ids <- unique(tmp$user_id)
  threshs <- vector(length=length(user_ids))
  for (i in 1:length(user_ids)) {
    tmpp <- tmp[user_id == user_ids[i]]
    l <- nrow(tmpp)
    f1s <- vector(length = l)
    for (k in 1:nrow(tmpp)){
      f1s[k] <- f1(tmpp$y, mat[1:l,k])
    }
  threshs[i] <- tmpp[,pred[which.max(f1s)]]
  }
  best_th[j] <- mean(threshs)
}

# # predictions are ordered descending, take the top k products
# user_ids <- unique(th$user_id)
# order_ids <- unique(th$order_id)
# baskets <- unique(th$round_basket)
# best_th <- vector(length=length(user_ids))
# for (j in 1:length(baskets)) {
#   tmp <- th[user_id==user_ids[j]]
#   l <- dim(tmp)[1]
#   f1s <- vector(length = l)
#   for (i in 1:dim(tmp)[1]){
#     f1s[i] <- f1(tmp$y, mat[1:l,i])
#   }
#   best_th[j] <- tmp[,pred[which.max(f1s)]]
# }

dt <- data.table(baskets,best_th)
th <- merge(th, dt, by.x="round_basket", by.y="baskets")
f1_score <- th[,.(f1score = f1(y,(pred>=best_th)*1)), user_id][,.(f1_mean=mean(f1score))]
f1_score

thresh_train <- data.table(user_id = user_ids, best_th = best_th)
th <- merge(th, thresh_train)
f1_score <- train_th[,.(f1score = f1(y,(pred>=best_th)*1)), user_id][,.(f1_mean=mean(f1score))]
f1_score

# Build Model to estimate threshold -----------------------------------

train_th <- th[,.(
  user_id=mean(user_id), 
  n_pp = .N, sd_pred=mean(sd(pred)), 
  m_pred = mean(pred), 
  max_pred=max(pred), 
  min_pred=min(pred), 
  y = mean(best_th), 
  pred_basket=mean(pred_basket),
  pnone = prod(1-pred)), order_id]

xtrain <- as.matrix(train_th[,-c("user_id", "order_id","y"), with=FALSE])
ytrain <- train_th$y

dtrain <- lgb.Dataset(xtrain, label=ytrain)

# Setting params for fitting
params_th <- list(
  objective           = "regression",
  num_leaves          = 4,
  learning_rate       = 0.03
)

valids <- list(train = dtrain)

folds[[1]] <- 1:10000
folds[[2]] <- 10001:20000
folds[[3]] <- 20001:30000
model_th <- lgb.cv(data = dtrain, params = params_th, nrounds = 670, folds=folds)

# get oof predictions for threshold model --------------
train_th$oof_pred <- NA
for (i in 1:length(folds)) {
  cat('\n\nTraining on fold', i,'...\n')
  cv_train <- train_th[-folds[[i]],]
  cv_val <- train_th[folds[[i]],]
  
  xtrain <- as.matrix(cv_train[,-c("user_id", "order_id", "y") ,with=FALSE])
  ytrain <- cv_train$y
  dtrain <- lgb.Dataset(xtrain,label=ytrain)
  
  xval <- data.matrix(cv_val[,-c("user_id", "order_id", "y"), with=FALSE])
  yval <- cv_val$y
  dval <- lgb.Dataset(xval,label=yval)
  
  bst <- lgb.train(params_th,dtrain,670,verbose=0) # first boosting iteration
  
  pred<-predict(bst,xval)
  train_th$oof_pred[folds[[i]]] <- pred
  
  rm(dtrain, dval, bst, cv_train, cv_val)
  gc()
}

rmse<-sqrt(mean((train_th$y-train_th$oof_pred)^2))
rmse
train_th %>% ggplot(aes(oof_pred,y))+geom_point()+geom_smooth(method="lm")

plot(unlist(model_th$record_evals$valid$l2$eval))
best_iter <- which.min(unlist(model_th$record_evals$valid$l2$eval))

th_model <- lgb.train(data=dtrain, params=params_th, nround=best_iter, valids=valids)

importance <- lgb.importance(th_model)
ggplot(importance,aes(y=Gain,x=reorder(Feature,Gain)))+geom_bar(stat="identity")+coord_flip()+theme(axis.text.y = element_text(hjust = 0))

th_test <- test[,.(user_id=user_id,order_id = order_id, pred=pred)][,':=' (pred_basket = sum(pred)), order_id][order(user_id,-pred)]
th_test<-th_test[,.(user_id=mean(user_id), n_pp = .N, sd_pred=mean(sd(pred)), pred_basket=mean(pred_basket)), order_id]

xtest <- as.matrix(th_test[,-c("user_id", "order_id"), with=FALSE])
dtest <- lgb.Dataset(xtrain)
pred_th <- predict(th_model, xtest)

th_test <- data.table(th_test[,.(user_id=user_id)], pred_th)





# Fit the Model to all training data -------------------------------------
xtrain <- as.matrix(train[,-c("user_id", "product_id", "order_id", "reordered"),with=FALSE])
ytrain <- train$reordered
dtrain <- lgb.Dataset(xtrain,label=ytrain)

valids <- list(train = dtrain)

model <- lgb.train(data = dtrain, params = params, nrounds = n_rounds, valids=valids)

importance <- lgb.importance(model)
#xgb.ggplot.importance(importance)+theme(axis.text.y = element_text(hjust = 0))
ggplot(importance,aes(y=Gain,x=reorder(Feature,Gain)))+geom_bar(stat="identity")+coord_flip()+theme(axis.text.y = element_text(hjust = 0))
ggplot(importance,aes(y=Gain,x=reorder(Feature,Feature)))+geom_bar(stat="identity")+coord_flip()+theme(axis.text.y = element_text(hjust = 0))

# Look at out of fold predictions ---------------------------------------------------------------
setkey(train, user_id)

train <- merge(train,train_info, all.x=TRUE)

train$pred <- predict(model,xtrain)
train <- train[order(user_id,-pred)]
train[,':=' (top = 1:.N), user_id]
train[,':=' (pred_basket = sum(pred)), user_id]


ttmp <- train[,.(sp=mean(pred_basket),sr=mean(sum_reordered)),user_id]
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
xtest <- as.matrix(test[,-c("user_id","order_id", "product_id"),with=FALSE])
test$pred <- predict(model, xtest)


# Threshold ---------------------------------------------------------------
test <- test[order(user_id,-pred)]
test[,':=' (top = 1:.N), user_id]
test[,':=' (pred_basket = sum(pred)), user_id]

test[pred_basket>30,':=' (adapted_basket = pred_basket+5)]
test[pred_basket<=30,':=' (adapted_basket = pred_basket)]
test[, ':=' (reordered=ifelse(top<=adapted_basket,1,0))]
#test[, ':=' (reordered=ifelse(top<=user_average_basket*user_reorder_ratio,1,0))]
#close_orders <- test %>% group_by(order_id) %>% summarize(m=mean(reordered),mx=max(reordered),s=sum(reordered>threshold)) %>% filter(between(m,0.9*threshold,1.1*threshold) & s <= 5 & mx <= 0.35) %>% select(order_id) %>% .[[1]]

test[,reordered:=(pred>=pred_th)*1]

# all reorderes to 1 -----------------------------------------------------
#test[user_id %in% reorder_users, ':=' (reordered=1)]

submission <- test[reordered==1,.(products = paste(product_id, collapse = " ")), order_id]

# add None to close orders -----------------------------------------------
#new_submission <- submission %>% mutate(products = ifelse(order_id %in% close_orders, str_c(products,'None', collapse = " "),products))



missing <- data.table(
  order_id = unique(test$order_id[!test$order_id %in% submission$order_id]),
  products = "None"
)

submission <- rbindlist(list(submission, missing))

# add none to close orders
tmpp <- test[order(-pred),.SD,order_id][,omp := 1-pred][,p_none := prod(omp), order_id]
none_orders <- tmpp[pred_basket < 1]$order_id
submission[(order_id %in% none_orders) & (products != "None"), products:=str_c(products,"None", sep=" ")]

fwrite(submission[order(order_id)], file = "submit.csv")


