#pragma once
#include "node_calculus.cpp"
#include <vector>
#include <stdio.h>

// Matrix is a 2D vector of NodePtrs
// tensor3 is a bunch of matrices, and well,
// you get where this is going
using Vector = std::vector<NodePtr>;
using Matrix = std::vector<Vector>;
using Tensor3 = std::vector<Matrix>;
using Tensor4 = std::vector<Tensor3>;

// Stuff I should get around to adding but I don't really
// need right now.
Vector addVector(Vector& v1, Vector& v2);
Vector addVector(Vector& v1, NumKin n);
Vector scaleVector(Vector& v1, NumKin s);

// More complicated functions that serve a proper purpose

Matrix addMatrix(const Matrix& A, const Matrix& B) {
    int w = A.size();
    int l = A[0].size();
    if (A.size() != B.size() || A[0].size() != B[0].size()) {
        throw std::invalid_argument("Can't broadcast together matrices of that size, dingus.");
    }

    Matrix sum(w, Vector(l, 0));
    for (int i=0; i < w; i++) {
        for (int j=0; j<l; j++) {
            sum[i][j] = addNodes(A[i][j], B[i][j]);
        }
    }

    return sum;
}

Matrix multMatrix(const Matrix& A, const Matrix& B) {
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

Tensor3 padTensor3(const Tensor3& input, int pad_h, int pad_w, const NodePtr& zero_node) {
    // input: C x H x W
    int C = input.size();
    int H = input[0].size();
    int W = input[0][0].size();

    int H_padded = H + 2 * pad_h;
    int W_padded = W + 2 * pad_w;

    Tensor3 output(C, Matrix(H_padded, Vector(W_padded, zero_node)));

    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                output[c][h + pad_h][w + pad_w] = input[c][h][w];
            }
        }
    }
    return output;
}

Tensor4 padTensor4(const Tensor4& input, int pad) {
    if (pad <= 0) return input;

    size_t padding = static_cast<size_t>(pad);

    size_t batch_size = input.size();
    size_t channels = input[0].size();
    size_t height = input[0][0].size();
    size_t width = input[0][0][0].size();

    size_t padded_height = height + 2 * padding;
    size_t padded_width = width + 2 * padding;

    Tensor4 padded(batch_size, Tensor3(channels, Matrix(padded_height, Vector(padded_width))));

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t y = 0; y < padded_height; ++y) {
                for (size_t x = 0; x < padded_width; ++x) {
                    if (y < padding ||
                        y >= padding + height ||
                        x < padding ||
                        x >= padding + width
                    ) {
                        // Pad with zeros
                        padded[b][c][y][x] = constantNode(0.0f);
                    } else {
                        padded[b][c][y][x] = input[b][c][y - padding][x - padding];
                    }
                }
            }
        }
    }
    return padded;
}

Matrix im2col(const Tensor3& input, int kernel_h, int kernel_w, int stride = 1) {
    int C = input.size();
    int H = input[0].size();
    int W = input[0][0].size();

    int out_h = (H - kernel_h) / stride + 1;
    int out_w = (W - kernel_w) / stride + 1;

    int patch_size = C * kernel_h * kernel_w;
    int num_patches = out_h * out_w;

    Matrix cols(patch_size, Vector(num_patches));

    int patch_idx = 0;
    for (int i = 0; i < out_h; ++i) {
        for (int j = 0; j < out_w; ++j) {
            int col_pos = 0;
            for (int c = 0; c < C; ++c) {
                for (int kh = 0; kh < kernel_h; ++kh) {
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        int h_idx = i * stride + kh;
                        int w_idx = j * stride + kw;
                        cols[col_pos][patch_idx] = input[c][h_idx][w_idx];
                        ++col_pos;
                    }
                }
            }
            ++patch_idx;
        }
    }
    return cols;
}

Matrix transpose(const Matrix& input) {
    if (input.empty()) {
        throw std::invalid_argument("What does it even mean to transpose a matrix of shape (0,0)?");
    }

    int rows = input.size();
    int cols = input[0].size();
    Matrix result(cols, Vector(rows));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[j][i] = input[i][j];
        }
    }

    return result;
}

Tensor4 conv2d(const Tensor4& input, const Tensor4& filters, int stride = 1, int padding = 0) {
    int batch_size = input.size();
    int out_channels = filters.size();
    int in_channels = input[0].size();
    int kernel_h = filters[0][0].size();
    int kernel_w = filters[0][0][0].size();

    // Pad input
    Tensor4 padded_input = padTensor4(input, padding);

    // Determine output spatial dimensions
    int out_h = (padded_input[0][0].size() - kernel_h) / stride + 1;
    int out_w = (padded_input[0][0][0].size() - kernel_w) / stride + 1;

    Tensor4 output(batch_size, Tensor3(out_channels, Matrix(out_h, Vector(out_w))));

    for (int b = 0; b < batch_size; ++b) {
        // im2col for the current input
        Matrix cols = im2col(padded_input[b], kernel_h, kernel_w, stride);  // shape: (out_h * out_w, in_channels * kernel_h * kernel_w)

        for (int oc = 0; oc < out_channels; ++oc) {
            // Flatten filter to a row vector
            Matrix filter_row(1);
            for (int ic = 0; ic < in_channels; ++ic) {
                for (int kh = 0; kh < kernel_h; ++kh) {
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        filter_row[0].push_back(filters[oc][ic][kh][kw]);
                    }
                }
            }

            // Matrix multiplication: (1 x K) * (K x N) = (1 x N)
            Matrix result = multMatrix(filter_row, transpose(cols));  // shape: (1, out_h * out_w)

            // Reshape result into output matrix
            for (int i = 0; i < out_h; ++i) {
                for (int j = 0; j < out_w; ++j) {
                    output[b][oc][i][j] = result[0][i * out_w + j];
                }
            }
        }
    }

    return output;
}

// Printouts
// TODO: Rework these so they aren't doodoo
void printTensor3(const Tensor3& tensor) {
    int C = tensor.size();
    for (int c = 0; c < C; ++c) {
        std::cout << "Channel " << c << ":\n";
        for (const auto& row : tensor[c]) {
            for (const auto& val : row) {
                std::cout << val->value << " ";
            }
            std::cout << "\n";
        }
        std::cout << "----\n";
    }
}

void printMatrix(const Matrix& mat) {
    for (const auto& row : mat) {
        for (const auto& val : row) {
            std::cout << val->value << " ";
        }
        std::cout << "\n";
    }
}

void printTensor4Values(const Tensor4& tensor) {
    for (size_t b = 0; b < tensor.size(); ++b) {
        std::cout << "Batch " << b << ":\n";
        for (size_t c = 0; c < tensor[b].size(); ++c) {
            std::cout << " Channel " << c << ":\n";
            for (size_t y = 0; y < tensor[b][c].size(); ++y) {
                for (size_t x = 0; x < tensor[b][c][y].size(); ++x) {
                    std::cout << tensor[b][c][y][x]->value << " ";
                }
                std::cout << "\n";
            }
        }
    }
}

