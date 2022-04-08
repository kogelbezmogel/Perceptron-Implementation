#include "../include/FunsContainer.h"
#include <iostream>

double signFun( double x );
double signFunGradiend(std::vector<double> x, double y, std::vector<double> w, int weight_number);


FunsContainer::FunsContainer() {
    std::cout << " FunsCon constructor \n";
    _act_fun_map.insert( {"sign", signFun} );
    _grad_fun_map.insert( {"signGradiend", signFunGradiend} );
}


double signFun( double x ) {
    return ( x > 0 ? 1 : -1);
}

double signFunGradiend(std::vector<double> x, double y, std::vector<double> w, int weight_number) {
        
    double ret = 0;
    double Li = 0;

    for( int i = 0; i < x.size(); ++i ) //counting error value if positive then classification is correct
        Li += x[i] * w[i];
    Li += w[ w.size() - 1 ] * 1;
    Li *= -y;

    if( Li < 0 ) //not correct classification then gradient value is not zero
        if ( weight_number != w.size() - 1 )
            ret = -y * x[weight_number];
        else
            ret = -y * 1; 

    return ret; 
}