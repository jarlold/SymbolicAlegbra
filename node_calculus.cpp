#pragma once
#include <iostream>
#include <cmath>
#include <memory>
#include <unordered_set>

typedef float NumKin;

enum Operation {
    MULT,
    ADD,
    POW,
    EXP,
    LOG,
    SIN,
    COS,
    TANH,
    DIV,
    RELU,
    NONE
};

struct Node {
    NumKin value = 0.0f;
    NumKin grad = 0.0f;
    Operation oper = NONE;
    std::shared_ptr<Node> lhs = nullptr;
    std::shared_ptr<Node> rhs = nullptr; // if the operation only takes
                                         // one input then this bad boy gets to stay null
};

// A man can only type so much
using NodePtr = std::shared_ptr<Node>;

// Should really be called "variable node" in hindsight.
NodePtr constantNode(NumKin v) {
    auto n = std::make_shared<Node>();
    n->value = v;
    return n;
}


// These are operations that take two expression-node-things
// and returns a new one, that performs the operation on them both
NodePtr addNodes(NodePtr lhs, NodePtr rhs) {
    auto n = std::make_shared<Node>();
    n->lhs = lhs;
    n->rhs = rhs;
    n->oper = ADD;
    return n;
}

NodePtr multNodes(NodePtr lhs, NodePtr rhs) {
    auto n = std::make_shared<Node>();
    n->lhs = lhs;
    n->rhs = rhs;
    n->oper = MULT;
    return n;
}

NodePtr powNode(NodePtr lhs, NodePtr rhs) {
    auto n = std::make_shared<Node>();
    n->lhs = lhs;
    n->rhs = rhs;
    n->oper = POW;
    return n;
}

NodePtr expNode(NodePtr lhs) {
    auto n = std::make_shared<Node>();
    n->lhs = lhs;
    n->oper = EXP;
    return n;
}

NodePtr logNode(NodePtr lhs) {
    auto n = std::make_shared<Node>();
    n->lhs = lhs;
    n->oper = LOG;
    return n;
}

NodePtr sinNode(NodePtr lhs) {
    auto n = std::make_shared<Node>();
    n->lhs = lhs;
    n->oper = SIN;
    return n;
}

NodePtr cosNode(NodePtr lhs) {
    auto n = std::make_shared<Node>();
    n->lhs = lhs;
    n->oper = COS;
    return n;
}

NodePtr tanhNode(NodePtr lhs) {
    auto n = std::make_shared<Node>();
    n->lhs = lhs;
    n->oper = TANH;
    return n;
}

NodePtr divNodes(NodePtr lhs, NodePtr rhs) {
    auto n = std::make_shared<Node>();
    n->lhs = lhs;
    n->rhs = rhs;
    n->oper = DIV;
    return n;
}

NodePtr reluNode(NodePtr lhs) {
    auto n = std::make_shared<Node>();
    n->lhs = lhs;
    n->oper = RELU;
    return n;
}

NodePtr sqrtNode(NodePtr lhs) {
    auto half = constantNode(0.5f);
    return powNode(lhs, half);
}

NodePtr negNode(NodePtr lhs) {
    auto minusOne = constantNode(-1.0f);
    return multNodes(minusOne, lhs);
}

// Forward pass with visited set
// This goes through and updates all the values in the nodes
void updateValue(NodePtr n, std::unordered_set<NodePtr>& visited) {

    // Make sure we don't re-use a node somehow.
    // This shouldn't get triggered but I'm pretty sure it does...
    if (!n || visited.count(n)) return;
    visited.insert(n);

    if (n->oper == NONE) return;

    if (n->lhs) updateValue(n->lhs, visited);
    if (n->rhs) updateValue(n->rhs, visited);

    switch (n->oper) {
        case MULT:
            n->value = n->lhs->value * n->rhs->value;
            break;
        case ADD:
            n->value = n->lhs->value + n->rhs->value;
            break;
        case POW:
            n->value = pow(n->lhs->value, n->rhs->value);
            break;
        case EXP:
            n->value = exp(n->lhs->value);
            break;
        case LOG:
            n->value = log(n->lhs->value);
            break;
        case SIN:
            n->value = sin(n->lhs->value);
            break;
        case COS:
            n->value = cos(n->lhs->value);
            break;
        case TANH:
            n->value = tanh(n->lhs->value);
            break;
        case DIV:
            n->value = n->lhs->value / n->rhs->value;
            break;
        case RELU:
            n->value = std::max(0.0f, n->lhs->value);
            break;
        default:
            break;
    }
}

NumKin forward(NodePtr root) {
    std::unordered_set<NodePtr> visited;
    updateValue(root, visited);
    return root->value;
}

// Backward pass with visited set
void updateBack(NodePtr n, std::unordered_set<NodePtr>& visited) {
    if (!n || visited.count(n)) return;
    visited.insert(n);

    if (n->oper == NONE) return;

    switch (n->oper) {
        case MULT:
            n->lhs->grad += n->rhs->value * n->grad;
            n->rhs->grad += n->lhs->value * n->grad;
            updateBack(n->lhs, visited);
            updateBack(n->rhs, visited);
            break;
        case ADD:
            n->lhs->grad += n->grad;
            n->rhs->grad += n->grad;
            updateBack(n->lhs, visited);
            updateBack(n->rhs, visited);
            break;
        case POW:
            if (n->lhs->value > 0) {
                n->lhs->grad += n->rhs->value * pow(n->lhs->value, n->rhs->value - 1) * n->grad;
                n->rhs->grad += log(n->lhs->value) * pow(n->lhs->value, n->rhs->value) * n->grad;
            }
            updateBack(n->lhs, visited);
            updateBack(n->rhs, visited);
            break;
        case EXP:
            n->lhs->grad += n->value * n->grad;
            updateBack(n->lhs, visited);
            break;
        case LOG:
            n->lhs->grad += (1 / n->lhs->value) * n->grad;
            updateBack(n->lhs, visited);
            break;
        case SIN:
            n->lhs->grad += cos(n->lhs->value) * n->grad;
            updateBack(n->lhs, visited);
            break;
        case COS:
            n->lhs->grad += -sin(n->lhs->value) * n->grad;
            updateBack(n->lhs, visited);
            break;
        case TANH: {
            NumKin t = tanh(n->lhs->value);
            n->lhs->grad += (1 - t * t) * n->grad;
            updateBack(n->lhs, visited);
            break;
        }
        case DIV: {
            NumKin a = n->lhs->value;
            NumKin b = n->rhs->value;
            n->lhs->grad += (1 / b) * n->grad;
            n->rhs->grad += (-a / (b * b)) * n->grad;
            updateBack(n->lhs, visited);
            updateBack(n->rhs, visited);
            break;
        }
        case RELU: {
            n->lhs->grad += (n->lhs->value > 0 ? 1.0f : 0.0f) * n->grad;
            updateBack(n->lhs, visited);
            break;
        }
        default:
            break;
    }
}

void backpropagation(NodePtr root) {
    root->grad = 1.0f;
    std::unordered_set<NodePtr> visited;
    updateBack(root, visited);
}

/*
int main() {
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

    return 0;
}
*/
