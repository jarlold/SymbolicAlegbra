#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <array>

typedef float NumKin;

enum Operation {
    MULT,
    ADD,
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
            n.children[1].grad = n.children[0].value * n.grad;
            n.children[0].grad = n.children[1].value * n.grad;
        break;
        
        case ADD:
            n.children[1].grad = 1 * n.grad;
            n.children[0].grad = 1 * n.grad;
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
    Node x_1 = constantNode(3.0, 1);
    Node x_2 = constantNode(4.0, 2);
    Node x_3 = multNodes(x_2, x_1, 3);
    Node x_4 = constantNode(5.0, 4);
    Node x_5 = addNodes(x_4, x_3, 5);
    Node x_6 = constantNode(2.0, 6);

    Node exp = multNodes(x_6, x_5, 7);


    NumKin out = forward(exp);
    printf("VALUE %f\r\n", out);
    updateBack(exp);
    printf("\r\n");
    doPrintGrad(exp);

}



