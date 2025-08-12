#pragma once
#include <vector>
#include <stdio.h>
#include <algorithm>
#include <memory>
#include <random>
#include <numeric>
#include "node_calculus.cpp"

// This is for stuff that other libraries would have
// that I don't. Maybe I should have called it utils..

std::vector<NumKin> range(int length) {
    std::vector<NumKin> nums(length);

    // I said random nums but actually...
    for (int i=0; i<length; i++) {
        nums[i] = i; 
    }

    return nums;
}

NumKin mean(const std::vector<NumKin>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0f) / v.size();
}

NumKin stddev(const std::vector<NumKin>& v, NumKin m) {
    NumKin sum = 0.0f;
    for (auto val : v) sum += (val - m) * (val - m);
    return std::sqrt(sum / v.size());
}

// Im gonna try out this inline thing i saw somewhere, but im pretty
// sure the compiler isn't dumb and knows how to do this on his own
inline float randomFloat(float min = -1.0f, float max = 1.0f) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    return dis(gen);
}

