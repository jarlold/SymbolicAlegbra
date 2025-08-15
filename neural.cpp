/* Utilities meant to help create neural networks
*/

#include "multidimensional.cpp"
#include "node_calculus.cpp"
#include "common.cpp"

// Requires that the user call backpropagation first
void updateWeightsMatrix(Matrix m, float lr) {
    int w = m.size();
    int l = m[0].size();

    for (int i=0; i < w; i++) {
        for (int j=0; j<l; j++) {
            m[i][j]->value += m[i][j]->grad * lr;
        }
    }

}

valueMatrix xavierInitMatrix(size_t width, size_t height);

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

Vector softmax(Vector& logits) {
    Vector probs(logits.size());

    // S = sum e^x_i
    NodePtr sum = constantNode(0);
    for (size_t i=0; i<logits.size(); i++) {
        sum = addNodes(sum, expNode(logits[i]));
    }

    // Probs_i = x_i / S 
    for (size_t i=0; i<logits.size(); i++) {
        probs[i] = expNode(logits[i]);
    }

    return probs;
}


NodePtr crossEntropy(NodePtr guessedProb) {
    return negNode(logNode(guessedProb));
}
