# dm2017
Data Mining Projects 2017


## Project 1:

In the function mapper we first tokenize the row (i.e. the value), so we obtain a list of shingles for each page (i.e. key). Since the mapper has access to only one page at the time, we create only a column of a signature matrix in a mapper. This column is initated with max possible values and then iteratively for each shingle we search for the smaller row index in the permutation. The permutations are done as described in the lecture where the prime numbers are fixed. The hashing search for the smaller row index in the permutation is vectorized because of the smaller computation times. Finally the Signature matrix column is separated in the bands where band ID is the key and page ID, band column and shingles of the given page are the value of the mapper.

The reducer takes as input the key which is the band ID and and the value contains all the coluns in that band, page IDs and the original shingles of the pages. We than hash the columns of the band so that the similar columns end up in the same bucket. The hashing is exactly the same as shown on the lecture slides. After we iterate all the buckets and check if the pages inside the same buckets have >.85 jaccard simillarity. This reduces the false positives to 0 and the computation is not expensive since there are only very few such cases.  

The parameters such as number of hash functions, nr. of bands and number of buckets used are all set empirically so that the number of false negatvies is 0 and number of false positives is not as small as possible. 

## Project 2:

In the mapper function we call transform(X) with X the chunk of data given to mapper and then svm_adam(X_transformed). The mapper then outputs the weights given by svm_adam as value and key is fixed to string "key". 

Reducer then collects all the weights from mappers and takes the mean of those to output the final weights. 

svm_adam() is the implementation of ADAM optimizer for hindge loss. The implemenentation is the same as pseudo code given in the lecture slides. The parameters such as alpha, beta1, beta2 are chosen to be default values of 1e-3, 0.9 and 0.999 respectively.

transform() is the function that projects the data to higher dimensional space by the inverse kernel trick described in the lecture. Here we use the gaussian kernel which means that weights are sampled from gaussian distirbution. Here we have empirically chosen standard deviation to be 4 and the number of new features to be 20000. 

## Project 3:

In this project, we have implemented the Loyd's k-means on the full data set. The only difference is the initialization of centroids. For this we use k-means++ approach described in paper by David Arthur and Sergei Vassilvitskii. 

Since we use k-means on the full data set, mapper just passes the data further and in the reducer is the whole k-means algorithm.

There are few help functions:
init() initializes the centroids with approach described by David Arthur and Sergei Vassilvitskii.
dist() computes the square of the l2 distance for each vector pair in the 2 input matrices.
compute_xtx() is the function that computes the sum of squares of each row in data matrix. It's useful to speed up the algorothm.
get_assignments() assigns each data point to a cluster.
get_means() recalculates the means of the clusters. 
stop() returns true if the algorithm converged.

The whole pipeline is implemented in the reducer and maximum number of iterations for k-means is fixed to 300 which is more than enough since empirically discovered, the algorithm stops after 50-60 iterations and sometimes even earlier.

## Project 4:

In this Project we have implemented the LinUCB algorithm as described on the slide 31 of the lecture slides on Bandtis problem. Since we had the user features and the article features, it made sense to start from this algorithm, since the previous proposed algorithms in the lectures did not take the context (i.e. user and article features) into account. The next step was to find good alpha parameter. This was one of the most difficult tasks, since the training data set provided was very small and the results obtained on the server differed a lot from those obtained on the local machine. Thus the usual grid-search approach did not work. In the mean time, we have found the paper on this problem and we there we have found the data set from which we extracted a random subset that was big enough to do the meaningful grid-search for alpha. This way we discovered that alpha of 0.16 was the best fit four our problem. Since this approach was shown to beat the hard baseline easily, there was no need for further development of the algorithm. Although we have implemented the hybrid version of the algorithm, we did not manage to find a good alpha in the reasonable amount of time.
