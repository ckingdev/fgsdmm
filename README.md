# FGSDMM
## Fast Gibbs Sampling for Dirichlet Multinomial Mixtures

This is an implementation of the collapsed Gibbs sampling algorithm introduced in
[A Dirichlet Multinomial Mixture Model-based Approach for
 Short Text Clustering (Yin and Wang, 2014)](http://dbgroup.cs.tsinghua.edu.cn/wangjy/papers/KDD14-GSDMM.pdf) using the optimizations discussed in [A Text Clustering Algorithm Using an Online Clustering
                                      Scheme for Initialization (Yin and Wang, 2016)](http://www.kdd.org/kdd2016/papers/files/rpp0617-yinAemb.pdf).

This is a hierarchical Bayesian model suitable for topic modelling over short texts.
The number of topics is bounded above by a hyperparameter, however, an optimization allows
for the complexity (time and space) to be approximately linear in the number of non-empty
clusters. Results of the above papers show that it is effective at finding the "true"
number of clusters in a corpus as long as the maximum number of clusters is chosen to
be greater than the true number of clusters.


## Warning

This is a work in progress and there will be breaking changes to the API.

The algorithm is correct currently and uses the optimization that allows for tracking only the
nonempty clusters, so it is efficient in that regard. It does not yet use the "FGSDMM+" optimization
that uses the DMM to sample the initial cluster assignments in an informed way.
