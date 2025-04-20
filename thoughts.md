# K means cluster
1. Build K means cluster using default k = 3
    1. Use random seed = 1 to generate a replicable random centroids
    2. Calculate distances using pytorch and reallocate centroids by means of distances for each points to their allocated centroids.
    3. Minimize the distances and variance to get optimal centroids.
    4. Output the labels clustered by the k means cluster
2. Iterate to find the elbow k

# SVM
1. Finding best fit support vector classifier
    1. Shuffle the data
    2. Calculate signed distances, reflecting the witch side each point is on, and it's euclidean distance to the SVC
    3. Define hinge loss
    4. Minimize the hinge loss function using iterations and gradient descent
    5. Output it's weights and bias
2. OVO classification (multiclass classification) implementation
    1. Add k means cluster output as datas labels
    2. Run SVC and do the classification
    3. Reallocate it's class by a voting system

# CAPM
1. Calculate R_m for multiple stocks within a SVM selected class
2. Run CAPM on R_m and R_i for each stock with R_m being the R_m calculated in step 1, and R_i being the return on each stock