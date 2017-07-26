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

f1e <- function(probs,k) {
  s <- length(probs)
  gt <- do.call("CJ",rep(list(c(0,1)),s))
  
  tmp <- gt*2-1
  tmp <- abs(-sweep(gt,2,probs, FUN="+")+1)
  
  ysel <- c(rep(1,k),rep(0,s-k))
  tmp <- apply(tmp,1,prod)
  res<-matrix(0,2,2^s)
  res[1,]<-apply(gt,1,FUN = function(x) f1q(x,ysel,1))
  res[2,]<-apply(gt,1,FUN = function(x) f1q(x,ysel,0))  
  res <- res %*% tmp
  dt <- data.table(k=k,none=which.max(res),f1=res[which.max(res)])
}

f1e2 <- function(probs,k) {
  s <- length(probs)
  gt <- do.call("CJ",rep(list(c(0,1)),s))
  
  tmp <- gt*2-1
  tmp <- abs(-sweep(gt,2,probs, FUN="+")+1)
  
  ysel <- c(rep(1,k),rep(0,s-k))
  tmp <- apply(tmp,1,prod)
  res<-matrix(0,2,2^s)
  res<-apply(gt,1,FUN = function(x) f1(x,ysel))
  res <- res %*% tmp
  dt <- data.table(k=k,none=which.max(res),f1=res[which.max(res)])
}



dofit <- function(probs){
  res <- data.table(k=integer(),none=integer(), f1=double())
  for (i in 0:length(probs)) {
    res <- rbindlist(list(res,f1e2(probs,i)))
  }
  res[f1==max(f1)]
}

ef1 <- function(probs){
  m <- length(probs)
  
  gt <- do.call("CJ",rep(list(c(0,1)),m))
  gt <- gt[2:nrow(gt)]
  gt <- data.table(gt, s=rowSums(gt))
  
  tmp <- gt[,1:m]*2-1
  tmp <- abs(-sweep(gt[,1:m],2,probs, FUN="+")+1)
  
  tmp <- data.table(tmp, p = apply(tmp,1,prod))
  
  gt <- data.table(gt, p = tmp[,p])
  
  p_is = matrix(NA, m,m)
  for (i in 1:m) {
    for (j in 1:m) {
      p_is[i,j] = sum(gt[gt[[j]]==1 & gt[,s]==i,p])
    }
  }
  w_sk = matrix(NA, m,m)
  for (s in 1:m) {
    for (k in 1:m) {
      w_sk[s,k]<-2/(s+k)
    }
  }
  p0 <- prod(1-probs)
  
  delta <- p_is %*% w_sk
  
  h <- vector(length=m)
  for (k in 1:m) {
    hi = c(rep(1,k),rep(0,m-k))
    h[k] = delta[,k] %*% hi
  }
  
  h <- c(p0,h)
  
  k <- which.max(h)-1
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
n_users <- 30000
all_train_users <- train_users[1:n_users]

all_users <- c(all_train_users, test_users)

rm(train_users, test_users)
gc()



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
users_per_fold <- 10000
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

calc_f1_rounds <- seq(110,200,10)
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

# do fixed thresholding first
train[, ':=' (pred_reordered = (oof_pred>0.20)*1)][,n_products_in_order := .N, order_id]
f1score<-train[,.(f1score = f1(reordered,pred_reordered)), order_id][,.(f1score=mean(f1score))]
f1score

# adjust thresholds
train<-train[order(user_id, -oof_pred)]
train[n_products_in_order<=2, c("k","none","f1") := dofit(oof_pred), order_id]
train[n_products_in_order<=2, k2 := ef1(oof_pred), order_id]

train[!is.na(k),':=' (pred_reordered = c(rep(1,mean(k)),rep(0,mean(n_products_in_order)-mean(k)))),order_id]

# train[n_products_in_order<=2,':=' (pred_reordered_old = c(rep(1,mean(k)),rep(0,mean(n_products_in_order)-mean(k)))),order_id]
# train[n_products_in_order<=2,':=' (pred_reordered_new = c(rep(1,mean(k2)),rep(0,mean(n_products_in_order)-mean(k2)))),order_id]
# 
# f1score<-train[n_products_in_order<=2,.(f1score = f1(reordered,pred_reordered_old)), order_id][,.(f1score=mean(f1score))]
# f1score
# 
# f1score<-train[n_products_in_order<=2,.(f1score = f1(reordered,pred_reordered_new)), order_id][,.(f1score=mean(f1score))]
# f1score

addnones <- train[k>0 & none==1,unique(order_id)]

submission1 <- train[pred_reordered==1,.(ypred = paste(product_id, collapse = " ")), order_id]
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
# add nones
submission1[order_id %in% addnones & ypred != "None", ypred:=paste(ypred, "None"), order_id]
submission2 <- rbindlist(list(submission2, missing2))

submission <- merge(submission1, submission2, by="order_id")

f1score<-submission[,.(f1score = f1m(y,ypred)), order_id][,.(f1score=mean(f1score))]
f1score


# Fit the User Product Model to all training data & predict test ---------------------------------
xtrain <- as.matrix(train[,-c("user_id", "product_id", "order_id", "reordered"),with=FALSE])
ytrain <- train$reordered
dtrain <- lgb.Dataset(xtrain,label=ytrain)

valids <- list(train = dtrain)

model <- lgb.train(data = dtrain, params = params, nrounds = best_iter, valids=valids)

rm(res, od, train, train_info, xtrain,ytrain,dtrain,valids, valid_orders, valid_users, y, pred, val_user_groups, val_users_random, all_users, all_train_users)
gc()

pred <- vector(length=nrow(test))
xtest <- as.matrix(test[,-c("user_id","order_id", "product_id"),with=FALSE])
pred <- predict(model, xtest)
rm(xtest)
gc()
test$pred <- pred
rm(pred)


test <- test[order(user_id,-pred)]

# do fixed thresholding first
test[, ':=' (pred_reordered = (pred>0.20)*1)][,n_products_in_order := .N, order_id]

# adjust thresholds
test[n_products_in_order<=10, c("k","none","f1") := dofit(pred), order_id]
test[!is.na(k),':=' (pred_reordered = c(rep(1,mean(k)),rep(0,mean(n_products_in_order)-mean(k)))),order_id]

addnones <- test[k>0 & none==1,unique(order_id)]

submission1 <- test[pred_reordered==1,.(products = paste(product_id, collapse = " ")), order_id]

missing1 <- data.table(
  order_id = unique(test$order_id[!test$order_id %in% submission1$order_id]),
  ypred = "None"
)

submission <- rbindlist(list(submission1, missing1))
# add nones
submission[order_id %in% addnones & products != "None", products:=paste(products, "None"), order_id] 

fwrite(submission[order(order_id)], file = "submit.csv")


