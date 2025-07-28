#include "node_calculus.cpp"
#include <vector>

// Matrix is a 2D vector of NodePtrs
using Matrix = std::vector<std::vector<NodePtr>>;

// Matrix multiplication: C = A × B
Matrix matmul(const Matrix& A, const Matrix& B) {
    size_t m = A.size();                 // Rows in A
    size_t k = A[0].size();              // Cols in A == Rows in B
    size_t n = B[0].size();              // Cols in B

    Matrix C(m, std::vector<NodePtr>(n));

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            NodePtr sum = multNodes(A[i][0], B[0][j]);
            for (size_t l = 1; l < k; ++l) {
                sum = addNodes(sum, multNodes(A[i][l], B[l][j]));
            }
            C[i][j] = sum;
        }
    }

    return C;
}



