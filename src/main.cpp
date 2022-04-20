#include <iostream>
#include <random>

#include "../include/perceptron.h"

int main() {

    Perceptron new_perc;

    std::default_random_engine generator;
    std::normal_distribution<double> group1( 4.0, 2 );
    std::normal_distribution<double> group2( 9.0, 2 );

    int group_count = 1000;

    MatData x_train;
    VecData y_train;

    for( int i = 0; i < group_count; ++i ) {
        x_train.push_back( std::vector<double> { group1(generator), group2(generator) } );
        y_train.push_back( -1 );
    }

    for( int i = 0; i < group_count; ++i ) {
        x_train.push_back( std::vector<double> { group2(generator), group1(generator) } );
        y_train.push_back( 1 );
    }

    new_perc.train(x_train, y_train);
    /*
    std::cout << "test-----------------------------------------------\n"; 

    int test_size = 30;
    MatData x_test1;
    MatData x_test2;
    for( int i = 0; i < test_size; ++i )
        x_test1.push_back( std::vector<double> { group1(generator), group2(generator) } );
    for( int i = 0; i < test_size; ++i )
        x_test2.push_back( std::vector<double> { group2(generator), group1(generator) } );

    VecData results1 = new_perc( x_test1 );
    VecData results2 = new_perc( x_test2 );
    for( double i : results1 )
        std::cout << i << " ";
    std::cout << "\n\n";
    for( double i : results2 )
        std::cout << i << "  ";
    std::cout << "\n";

    std::cout << "test-end-------------------------------------------\n\n"; 
    */

    FILE* file = fopen("data.dat", "w");
    for( int i = 0; i < x_train.size(); ++i ) {
        fprintf(file, "%15f %15f %15f", x_train[i][0], x_train[i][1], y_train[i]);
        if( i < x_train.size() - 1)
            fprintf(file, "\n");
    }
    fclose(file);    

    std::vector<double> vec = new_perc.weights();
    FILE* file2 = fopen("line.dat", "w");
        fprintf( file2, "%15f %15f %f", vec[0], vec[1], vec[2] );
    fclose(file2);    

    std::cout << new_perc;

return 0;
}