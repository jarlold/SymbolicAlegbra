#include "node_calculus.cpp"
#include "common.cpp"
#include "multidimensional.cpp"

void test1() {
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
    auto C = multMatrix(A, B);

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
}

void test2() {
    auto a = constantNode(3.0f);
    auto b = constantNode(-1.5f);
    auto c = constantNode(0.0f);
    auto d = constantNode(2.2f);
    auto e = constantNode(4.5f);
    auto f = constantNode(-0.7f);
    auto g = constantNode(1.3f);

    // Unary ops
    auto sin_a = sinNode(a);
    auto cos_b = cosNode(b);
    auto tanh_c = tanhNode(c);
    auto relu_b = reluNode(b);
    auto neg_d = negNode(d);
    auto exp_e = expNode(e);
    auto log_d = logNode(d);
    auto sqrt_g = sqrtNode(g);
    auto neg_f = negNode(f);
    auto relu_c = reluNode(c);

    // Binary ops chaining
    auto add1 = addNodes(sin_a, cos_b);
    auto mul1 = multNodes(add1, tanh_c);
    auto pow1 = powNode(mul1, constantNode(2.0f));
    auto pow2 = powNode(a, b);
    auto pow3 = powNode(sqrt_g, f);  // sqrt(g)^f

    auto add2 = addNodes(pow1, pow2);
    auto add3 = addNodes(add2, pow3);

    // More unary inside binary
    auto log_mul1 = logNode(mul1);   // may be invalid if mul1 <= 0
    auto exp_neg_d = expNode(neg_d);
    auto div2 = divNodes(log_mul1, exp_neg_d);

    auto neg_pow2 = negNode(pow2);
    auto sum_all1 = addNodes(add3, div2);
    auto sum_all = addNodes(sum_all1, neg_pow2);

    // Deep nested relus and negations
    auto relu_sum = reluNode(sum_all);
    auto neg_relu_sum = negNode(relu_sum);
    auto relu_neg_relu_sum = reluNode(neg_relu_sum);

    // Divide by zero cases and chaining divisions
    auto div3 = divNodes(sum_all, relu_c);  // relu_c = 0 for c=0.0
    auto div4 = divNodes(div3, relu_b);     // relu_b = 0 for negative b

    // Final complex expression
    auto sin_div4 = sinNode(div4);
    auto tanh_neg_f = tanhNode(neg_f);
    auto cos_pow3 = cosNode(pow3);

    auto mul_final = multNodes(sin_div4, tanh_neg_f);
    auto add_final1 = addNodes(div4, exp_e);
    auto add_final2 = addNodes(add_final1, negNode(log_d));
    auto add_final3 = addNodes(add_final2, mul_final);
    auto final_expr = addNodes(add_final3, cos_pow3);

    // Forward and backward
    forward(final_expr);
    backpropagation(final_expr);

    // Print grads and values
    std::cout << "a grad: " << a->grad << ", value: " << a->value << "\n";
    std::cout << "b grad: " << b->grad << ", value: " << b->value << "\n";
    std::cout << "c grad: " << c->grad << ", value: " << c->value << "\n";
    std::cout << "d grad: " << d->grad << ", value: " << d->value << "\n";
    std::cout << "e grad: " << e->grad << ", value: " << e->value << "\n";
    std::cout << "f grad: " << f->grad << ", value: " << f->value << "\n";
    std::cout << "g grad: " << g->grad << ", value: " << g->value << "\n";
}

void test3() {
    // Create zero node
    NodePtr zero_node = constantNode(0.0f);

    // Create Tensor3 with 1 channel, 3x3 grid, values 1..9 as constantNodes
    int C = 1, H = 3, W = 3;
    Tensor3 input(C, Matrix(H, Vector(W)));

    int val = 1;
    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                input[c][h][w] = constantNode(static_cast<NumKin>(val++));
            }
        }
    }

    std::cout << "Original Tensor3 values:\n";
    for (const auto& row : input[0]) {
        for (const auto& n : row) {
            std::cout << n->value << " ";
        }
        std::cout << "\n";
    }

    // Pad tensor with 1 zero border
    Tensor3 padded = padTensor3(input, 1, 1, zero_node);

    std::cout << "\nPadded Tensor3 values:\n";
    for (const auto& row : padded[0]) {
        for (const auto& n : row) {
            std::cout << n->value << " ";
        }
        std::cout << "\n";
    }

    // im2col with 2x2 kernel, stride 1
    int kernel_h = 2, kernel_w = 2;
    Matrix cols = im2col(padded, kernel_h, kernel_w);

    // For each patch (column), create a sum node by adding all nodes in the patch
    int num_patches = cols[0].size();
    std::vector<NodePtr> patch_sums(num_patches);

    for (int p = 0; p < num_patches; ++p) {
        NodePtr sum_node = constantNode(0.0f);
        for (int i = 0; i < (int)cols.size(); ++i) {
            sum_node = addNodes(sum_node, cols[i][p]);
        }
        patch_sums[p] = sum_node;
    }

    // Forward pass: update values for all patch sums
    for (auto& node : patch_sums) {
        std::unordered_set<NodePtr> visited;
        updateValue(node, visited);
    }

    std::cout << "\nPatch sums (forward pass):\n";
    for (auto& node : patch_sums) {
        std::cout << node->value << " ";
    }
    std::cout << "\n";

    // Set gradient 1 for all patch sums and backpropagate
    for (auto& node : patch_sums) {
        node->grad = 1.0f;
        std::unordered_set<NodePtr> visited;
        updateBack(node, visited);
    }

    // Print gradients for original input nodes to see backprop through padding + im2col
    std::cout << "\nGradients on original input nodes after backpropagation:\n";
    for (const auto& row : input[0]) {
        for (const auto& n : row) {
            std::cout << n->grad << " ";
        }
        std::cout << "\n";
    }
}

void test4() {
    // 1 input, 1 channel, 3x3
    Tensor4 input(1, Tensor3(1, Matrix(3, Vector(3))));
    // 1 filter, 1 channel, 2x2
    Tensor4 filters(1, Tensor3(1, Matrix(2, Vector(2))));

    // Fill input with constant nodes (values 1 to 9)
    int val = 1;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            input[0][0][i][j] = constantNode(val++);

    // Fill filter with constant nodes (values 1, 0, -1, 0)
    std::vector<float> filter_vals = {1, 0, -1, 0};
    int idx = 0;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            filters[0][0][i][j] = constantNode(filter_vals[idx++]);

    // Run conv2d (no padding, stride=1)
    Tensor4 out = conv2d(input, filters, 1, 0);

    // Take one output scalar and backprop from it
    NodePtr output_node = out[0][0][0][0];  // First element
    backpropagation(output_node);

    // Print input gradients
    std::cout << "Input gradients:\n";
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            std::cout << input[0][0][i][j]->grad << " ";
        }
        std::cout << "\n";
    }

    // Print filter gradients
    std::cout << "\nFilter gradients:\n";
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            std::cout << filters[0][0][i][j]->grad << " ";
        }
        std::cout << "\n";
    }

}

void test5() {
    // 1 input, 1 channel, 5x5
    Tensor4 input(1, Tensor3(1, Matrix(5, Vector(5))));
    // 1 filter, 1 channel, 3x3
    Tensor4 filters(1, Tensor3(1, Matrix(3, Vector(3))));

    // Fill input with constant nodes (values 1 to 25)
    int val = 1;
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 5; ++j)
            input[0][0][i][j] = constantNode(val++);

    // Fill filter with constant nodes (values for Sobel-like filter)
    std::vector<float> filter_vals = {
        1, 0, -1,
        1, 0, -1,
        1, 0, -1
    };
    int idx = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            filters[0][0][i][j] = constantNode(filter_vals[idx++]);

    // Run conv2d (no padding, stride=1)
    Tensor4 out = conv2d(input, filters, 1, 0);

    // Take one output scalar and backprop from it
    NodePtr output_node = out[0][0][0][0];  // First element of output
    backpropagation(output_node);

    // Print input gradients
    std::cout << "Input gradients:\n";
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            std::cout << input[0][0][i][j]->grad << " ";
        }
        std::cout << "\n";
    }

    // Print filter gradients
    std::cout << "\nFilter gradients:\n";
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            std::cout << filters[0][0][i][j]->grad << " ";
        }
        std::cout << "\n";
    }
}


void sgd_test1() {
    // This test does gradient descent on A SINGLE VALUE
    // at at time, using NO INPUT NORMALIZATION
    // matching the values to pytorch really serves as 
    // our most basic sanity check.

    // Generate a bunch of data
    std::vector<NumKin> x = range(200);
    std::vector<NumKin> y(200);
    for (int i=0; i<200; i++) {
        y[i] = 5.2 * x[i] + 22.8;
    }

    // Then we'll build our model
    NodePtr two = constantNode(2.0);
    NodePtr w = constantNode(1.0);
    NodePtr b = constantNode(1.0);
    NodePtr xi = constantNode(0.0);
    NodePtr yi = constantNode(0.0);
    NodePtr pred = addNodes(multNodes(xi, w), b);
    NodePtr loss = powNode(addNodes(yi, negNode(pred)), two);

    // Then we'll just try following along the gradient until its better
    for (int i = 0; i < 10; i++) {
        float numLoss;
        int index = i; // rand() % 200;
        xi->value = x[index];
        yi->value = y[index];
        resetGraph(loss);
        numLoss = forward(loss);

        if (numLoss <= 1.0) break;


        backpropagation(loss);

        printf(
            "w Grad %f\r\n"
            "b Grad %f\r\n"
            "w Val  %f\r\n"
            "b Val  %f\r\n"
            "Loss   %f\r\n\r\n",
            w->grad,
            b->grad,
            w->value,
            b->value,
            numLoss
        );

        w->value -= w->grad  * 0.0001;
        b->value -= b->grad * 0.0001;


    }
}


int main() {
    test1();
    test2();
    test3();
    test4();
    test5();
    sgd_test1();
    return 0;
}

