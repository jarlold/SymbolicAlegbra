#include "node_calculus.cpp"
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

int main() {
    test1();
    test2();
    return 0;
}

