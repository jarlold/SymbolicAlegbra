#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <array>
#include <cmath>

typedef float NumKin;

enum Operation {
    MULT,
    ADD,
    POW,
    NONE
};

struct Node {
  NumKin value;
  NumKin grad;
  Operation oper;
  std::vector<Node> children;
  int id;
};

struct Expression {
    std::vector<Node> nodes;
    Node& root;
};

Node addNodes(Node& n1, Node& n2, int id) {
    Node n3;
    n3.children.push_back(n1);
    n3.children.push_back(n2);
    n3.oper = ADD;
    n3.grad = 1;
    n3.id = id;
    return n3;
}

Node multNodes(Node& n1, Node& n2, int id) {
    Node n3;
    n3.children.push_back(n1);
    n3.children.push_back(n2);
    n3.oper = MULT;
    n3.grad = 1;
    n3.id = id;
    return n3;
}


Node powNode(Node& n1, Node& n2, int id) {
    // No taking powers like x^x
    if (n2.oper != NONE) {
        throw std::invalid_argument("Can't take things to the power of a function just yet.");
    }

    Node n3;
    n3.children.push_back(n1);
    n3.children.push_back(n2);
    n3.oper = POW;
    n3.grad = 1;
    n3.id = id;

    return n3;
}

Node constantNode(NumKin v, int id) {
    Node n;
    n.oper = NONE;
    n.grad = 1;
    n.value = v;
    n.id = id;
    return n;
}

void updateValue(Node& n) {
    if (n.oper == NONE) {
        return;
    }

    updateValue(n.children[0]);
    updateValue(n.children[1]);

    switch (n.oper) {
        case MULT:
            n.value = n.children[0].value * n.children[1].value;
        break;
        
        case ADD:
            n.value = n.children[0].value + n.children[1].value;
        break;

        case POW:
            n.value = pow(n.children[0].value, n.children[1].value);
        break;
        
        case NONE:
          // This shouldn't happen don't do this
        break;
    }
}

NumKin forward(Node& root) {
    updateValue(root);
    return root.value;
}

void updateBack(Node& n) {
    // This function goes through the syntax tree updating the 
    // grad on all the children.

    if (n.oper == NONE) {
        return;
    }

    switch (n.oper) {
        case MULT:
            n.children[0].grad = n.children[1].value * n.grad;
            n.children[1].grad = n.children[0].value * n.grad;
        break;
        
        case ADD:
            n.children[0].grad = 1 * n.grad;
            n.children[1].grad = 1 * n.grad;
        break;

        case POW:
            // (x^8)` = 8(x^7)
            // This won't work for taking things to a power that isn't NumberKin, to solve that I've
            // forbidden it.
            n.children[0].grad =
                n.children[1].value * pow( n.children[0].value, n.children[1].value-1) * n.grad;
        break;
        
        case NONE:
          // This shouldn't happen don't do this
        break;
    }

    updateBack(n.children[0]);
    updateBack(n.children[1]);
}

void backpropogation(Node& root) {
    root.grad = 1;
}

void doPrintGrad(Node& root) {
    if (root.oper == NONE) {
        printf("%d: Grad %f, Value %f (const)\r\n", root.id, root.grad, root.value);
    } else {
        printf("%d: Grad %f, Value %f\r\n", root.id, root.grad, root.value);
        doPrintGrad(root.children[0]);
        doPrintGrad(root.children[1]);
    }
}


int main() {
    Node x_1 = constantNode(5.0, 1);
    Node x_2 = constantNode(4.0, 2);
    Node exp = powNode(x_1, x_2, 3);

    NumKin out = forward(exp);
    printf("VALUE %f\r\n", out);
    updateBack(exp);
    printf("\r\n");
    doPrintGrad(exp);
}



