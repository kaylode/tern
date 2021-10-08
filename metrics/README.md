# Metric 

## Descriptions
Name | Description
--- | --- 
Nearest Neighbor (**NN**) | Given a query, it is the precision at the first object of the retrieved list.
First Tier (**FT**) | Given a query, it is the precision when C objects have been retrieved, where C is the number of relevant objects to the query.
Second Tier (**ST**) | Given a query, it is the precision when 2*C objects have been retrieved, where C is the number of relevant objects to the query.
Recall@K (**R@K**) | This metric gives how many actual relevant results were shown out of all actual relevant results for the query.
F1@K (**F1@K**) | This is a combined metric that incorporates both Precision@K and Recall@K by taking their harmonic mean.
Mean Average Precision (**MAP@K**) |  Given a query, its average precision is the average of all precision values computed in each relevant object in the retrieved list. Given several queries, the mean average precision is the mean of average precision of each query.

## References
- https://amitness.com/2020/08/information-retrieval-evaluation/
