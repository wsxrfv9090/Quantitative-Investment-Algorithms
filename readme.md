# Usage
1. Set absolute directory to your clone destinations.
e.g: `d:\Important Files\Repositories\Quantitative-Investment-Algorithms'`

# To do list

- [ ] 1. CAPM algorithms
- [x]   1. Get the test data
- [x]   2. Preprocess test data
- [x]   3. Compute R_i, R_m, R_f
- [x]   4. Compute Excessive Returns for both the stock and market
- [ ]   Improvements to be made:
- [ ]   1. Eliminate redundant work
- [ ]   2. Swap out statsmodels for a closed‑form beta: Use vectorized closed‑form formulas
- [ ]   3. Parallelize at the “file” level: with `multiprocessing.Pool` (or `joblib`) rather than threads.
- [ ]   4. When (and when not) to consider GPU
- [ ]   5. Isolate different code parts to make it more readable.
- [ ] 2. Get the data
- [x] 3. K means clustering to get labels for svm
- [x]   1. Randomly choose k points as centroids
- [x]   2. Assign points to their closest centroids
- [x]   3. Calculate the mean of each cluster as new centroids
- [x]   4. Repeat until the clusters doesn't change within tolerance
- [x]   5. Evaluate the clustering with the total variation as evaluator
- [x]   6. Repeat to find the minimum total variation
- [x]   7. Elbow plot
- [ ] 4. SVM using pytorch algorithm, this draws the line for binary classification
- [ ]    Finding best fit support vector classifier
- [ ]      1. Shuffle the data
- [ ]      2. Calculate signed distances, reflecting the witch side each point is on, and it's euclidean distance to the SVC
- [ ]      3. Define hinge loss
- [ ]      4. Minimize the hinge loss function using iterations and gradient descent
- [ ]      5. Output it's weights and bias
- [ ] 5. OVO using pytorch algorithm, do one by one voting system, this uses the SVM function for each 2 clusters clustered by the k means cluster
- [ ]    OVO classification (multiclass classification) implementation
- [ ]      1. Add k means cluster output as datas labels
- [ ]      2. Run SVC and do the classification
- [ ]      3. Reallocate it's class by a voting system
- [ ] 6. (Optional) if the data is not linearly seperatable by svm, inplement kernel tricks using polynomial or other method to seperate it.
- [ ] 7. Choose class index as risk free rate then calculate alpha and beta for each stock within the class
- [ ] 8. Evaluate combinations
