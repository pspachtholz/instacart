### Which products are bought together?
```{r warning=FALSE}
library(arules)
tmp <- order_products %>% select(order_id,product_id) %>% mutate(product_id = as.factor(product_id)) %>% as.data.frame()

tmp <- tmp[1:100,] # only look at first 1,000 products to avoid memory issues
transactions <- as(split(tmp[,"product_id"], tmp[,"order_id"]),"transactions")

inspect(head(transactions,4))


rules <- apriori(transactions) 
rules_conf <- sort(rules, by="confidence", decreasing=TRUE) # 'high-confidence' rules.
inspect(head(rules_conf))

```