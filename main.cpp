#include <iostream>
#include <random>

#include "perceptron.h"

int main() {

    std::cout << "\nPoczatek\n";

    Perceptron new_perc;

    std::default_random_engine generator;
    std::normal_distribution<double> group1( 0.0, 3.0 );
    std::normal_distribution<double> group2( 9.0, 3.0 );

    int group_count = 600;

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

    int test_size = 100;
    MatData x_test1;
    for( int i = 0; i < test_size; ++i )
        x_test1.push_back( std::vector<double> { group1(generator), group2(generator) } );
    for( int i = 0; i < test_size; ++i )
        x_test1.push_back( std::vector<double> { group2(generator), group1(generator) } );

    VecData results = new_perc( x_test1 );
    for( double i : results )
        std::cout << i << " ";

    FILE* file = fopen("data.dat", "w");
    for( int i = 0; i < x_train.size(); ++i ) {
        fprintf(file, "%15f %15f %d", x_train[i][0], x_train[i][1], y_train[i]);
        if( i < x_train.size() - 1)
            fprintf(file, "\n");
    }
    fclose(file);    

    std::vector<double> vec = new_perc.wages();
    FILE* file2 = fopen("line.dat", "w");
        fprintf( file2, "%15f %15f %f", vec[0], vec[1], new_perc.bias() );
    fclose(file2);    

    //std::cout << new_perc;
    std::cout << "\nKoniec\n";

return 0;
}