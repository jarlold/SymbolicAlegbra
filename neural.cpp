/* Utilities meant to help create neural networks
*/

#include "multidimensional.cpp"
#include "node_calculus.cpp"
#include "common.cpp"


// Requires that the user call backpropagation first
void updateWeightsMatrix(Matrix m, float lr) {
    int l = m.size();
    int w = m[0].size();

    for (int i=0; i < w; i++) {
        for (int j=0; j<l; j++) {
            m[i][j]->value += m[i][j]->grad * lr;
        }
    }

}


void xavierInitializeMatrix(Matrix m, int depth);


