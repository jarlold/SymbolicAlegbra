/*  UNBATCHED loss functions can go here. The batched version belong in
    neural.cpp because that's what they're used for.
*/


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


/*
// This should do the no-softmax thing fine but I haven't tested it
// properly.
NodePtr crossEntropy(Vector& predictions, size_t trueLabel) { 
    NodePtr loss = negNode(logNode(predictions[trueLabel]));
    return loss;
}
*/


NodePtr crossEntropyLogits(Vector& logits, size_t trueLabel) {
    // Reminder to fix this thing later
    throw std::invalid_argument(
        "You forgot to finish implementing crossentropy."
        "You also should look up a better error type once you get internet again."
    );

    // Start by doing softmax
    // I think I fucked up this implementation it doesn't
    // match pytorch.
    Vector exps(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        exps[i] = expNode(logits[i]);
    }

    NodePtr sumExp = exps[0];
    for (size_t i = 1; i < exps.size(); ++i) {
        sumExp = addNodes(sumExp, exps[i]);
    }

    NodePtr probTrue = divNodes(exps[trueLabel], sumExp);

    // Then actually do cross entropy
    NodePtr loss = negNode(logNode(probTrue));

    return loss;
}



