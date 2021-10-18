def my_PCA(X, n_comp=2):
    # Covariasion matrix
    X_cov_mat = np.cov(X)

    # Eigenvalues and eigenvectors
    X_eig_val_cov, X_eig_vec_cov = np.linalg.eig(X_cov_mat)

    # Make a list of (eigenvalue, eigenvector) tuples
    X_eig_pairs = [(np.abs(X_eig_val_cov[i]), X_eig_vec_cov[:, i]) for i in range(len(X_eig_val_cov))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    X_eig_pairs.sort(key=lambda x: x[0], reverse=True)

    X_matrix_w = np.hstack((X_eig_pairs[0][1].reshape(2624, 1), X_eig_pairs[1][1].reshape(2624, 1)))
    for i in range(2, n_comp):
        X_matrix_w = np.hstack((X_matrix_w, X_eig_pairs[1][1].reshape(2624, 1)))
    print('Matrix W:\n', X_matrix_w)

    X_transformed = X_matrix_w.T.dot(X)
    return X_transformed