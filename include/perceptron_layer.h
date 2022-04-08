#ifndef __PERCEPTRON_LAYER__
#define __PERCEPTRON_LAYER__

#include <functional>
#include <iostream>
#include <vector>
#include "perceptron.h"


class SingleLayerPerceptron {

    private:
        std::vector< Perceptron > _preceptrons;

    public:
        SingleLayerPerceptron();
};

#endif //__PERCEPTRON_LAYER__