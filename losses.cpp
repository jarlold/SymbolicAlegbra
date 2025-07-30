#pragma once
#include "multidimensional.cpp"
#include "node_calculus.cpp"
#include "common.cpp"

NodePtr meanSquaredError(Vector& predictions, Vector& groundTruth) {
    NodePtr loss = constantNode(0);
    NodePtr two = constantNode(2); // dont judge me
    int len = groundTruth.size();
    NodePtr lenN = constantNode(len);

    for (int i=0; i<len; i++) {
        NodePtr iloss = powNode(addNodes(predictions[i], negNode(groundTruth[i])), two);
        loss = addNodes(loss, iloss);
    }

    loss = divNodes(loss,lenN);
    return loss;
}


NodePtr crossEntropy(Vector& predictions, size_t trueLabel) { 
    NodePtr loss = negNode(logNode(predictions[trueLabel]));
    return loss;
}
