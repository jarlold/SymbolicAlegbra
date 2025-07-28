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

int main() {
    // Create a 5x4 matrix A with values: sin(0.1 * i * j + 1)
    std::vector<std::vector<NodePtr>> A(5, std::vector<NodePtr>(4));
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 4; ++j)
            A[i][j] = constantNode(std::sin(0.1f * i * j + 1.0f));

    // Create a 4x8 matrix B with values: cos(0.2 * i - 0.3 * j + 2)
    std::vector<std::vector<NodePtr>> B(4, std::vector<NodePtr>(8));
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 8; ++j)
            B[i][j] = constantNode(std::cos(0.2f * i - 0.3f * j + 2.0f));

    // Matrix multiplication: (5x4) * (4x8) = (5x8)
    auto C = matmul(A, B);

    // Apply tanh to each output node and sum to create scalar loss
    NodePtr loss = tanhNode(C[0][0]);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 8; ++j) {
            if (i != 0 || j != 0) {
                loss = addNodes(loss, tanhNode(C[i][j]));
            }
        }
    }

    // Run forward and backward passes
    forward(loss);
    backpropagation(loss);

    // Print gradients of A
    std::cout << "Gradients of A:\n";
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << A[i][j]->grad << "\t";
        }
        std::cout << "\n";
    }

    // Print gradients of B
    std::cout << "\nGradients of B:\n";
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 8; ++j) {
            std::cout << B[i][j]->grad << "\t";
        }
        std::cout << "\n";
    }

    return 0;
}

