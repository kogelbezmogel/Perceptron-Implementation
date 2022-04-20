#include "../include/FunsContainer.h"
#include <iostream>

double signFun( double x );
double linearFun( double x );
double signFunGradiend(std::vector<double> x, double y, std::vector<double> w, int weight_number);
double linearFunGradiend(std::vector<double> x, double y, std::vector<double> w, int weight_number);


FunsContainer::FunsContainer() {
    //std::cout << " FunsCon constructor \n";
    _act_fun_map.insert( {"sign", signFun} );
    _act_fun_map.insert( {"linear", linearFun} );
    _grad_fun_map.insert( {"signGradiend", signFunGradiend} );
    _grad_fun_map.insert( {"linearGradiend", linearFunGradiend} ); 
}

double signFun( double x ) {
    return ( x >= 0 ? 1 : -1);
}

double linearFun( double x ) {
    return x;
}

double signFunGradiend(std::vector<double> x, double y, std::vector<double> w, int weight_number) {
    //sth is not working properly    
    double ret = 0;
    double Li = 0;

    for( int i = 0; i < x.size(); ++i ) //counting error value if positive then classification is correct
        Li += x[i] * w[i];
    Li += w[ w.size() - 1 ];
    Li *= -y;

    if( Li > 0 ) { //not correct classification then gradient value is not zero
        if ( weight_number < w.size() - 1 )
            ret = -y * x[weight_number];
        else
            ret = -y; 
    }
    std::cout << "e_y: " << y << "  Li: " << Li << "  grd: " << ret << "\n";
    return ret; 
}

double linearFunGradiend(std::vector<double> x, double y, std::vector<double> w, int weight_number) {  
    double ret = 0;
    double calculated_y = 0;
    double error;
    
    for( int i = 0; i < x.size(); ++i )
        calculated_y += x[i] * w[i];
    calculated_y += w[ w.size() - 1 ];

    error = y - calculated_y;

    if( weight_number < x.size() )
        ret = - error * x[weight_number];
    else
        ret = - error; 

    return ret; 
}