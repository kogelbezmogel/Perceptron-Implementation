#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <ctime>

#include "../include/perceptron.h"
#include "../include/data_generation.h"
#include "../include/data_frame.h"

int main() {

    // genereting data to csv file -------------------------------------- 
    DataGeneratorC model = DataGeneratorC();
    model.set_first_gruop_params(8, 1, 8, 1);
    model.set_second_group_params(10, 1, 10 ,1);
    model(4000, "./data_files/data1.csv");

    DataGeneratorR modelR = DataGeneratorR(3, -1);
    modelR(0, 50, "./data_files/linreg.csv");
    // end of data generation 


    // loading data from csv file ----------------------------------------
    std::ifstream file;
    std::string line;

    file.open("./data_files/data1.csv");
    getline(file, line);
    std::vector< std::string > header_row;
    std::vector< double > row;
    Mat results;
    std::string word;

    std::stringstream ss(line);
    const std::stringstream::pos_type start{ ss.tellg( ) };

    while( getline(ss, word, ',') ) {
        header_row.push_back(word);
    }
    
    while ( getline(file, line) ) {
        row.clear();
        ss.clear();
        ss.seekg(start);
        ss.str(line);

        while( getline(ss, word, ',') ) { row.push_back( std::stof(word) ); }

        results.push_back(row);
    }    
    file.close();
    // end of loading data
    

    // extracting randomly some part of data to train an to test ---------
    int min_id = 0;
    int max_id = results.size();
    int elements_amount = results[0].size();
    double train_size = 0.7;
    int train_chunk_data_size = train_size * max_id;

    std::mt19937 engine;
    int s = std::time(0);
    //std::cout << "Seed: " << s << "\n";
    engine.seed(s);
    std::shuffle(results.begin(), results.end(), engine);

    Mat train_data;
    Mat test_data;
    int i = 0;
    for(; i < train_chunk_data_size; i++) 
        train_data.push_back(results[i]);
    for(; i < max_id; i++)
        test_data.push_back(results[i]);
    
    //std::cout << "train: " << train_data.size() << "\ntest:  " << test_data.size() << "\n";

    Mat x_train, y_train, x_test, y_test;
    std::vector<double> temp;

    for( auto row : train_data ) {
        for(int j = 0; j < elements_amount; j++) {
            if( j != elements_amount-1 )
                temp.push_back(row[j]);
            else if( j == elements_amount-1 )
                y_train.push_back( std::vector<double>( {row[j]} ) );
        }
        x_train.push_back(temp);
        temp.clear();
    }

    for( auto row : test_data ) {
        for(int j = 0; j < elements_amount; j++) {
            if( j != elements_amount-1 )
                temp.push_back(row[j]);
            else if( j == elements_amount-1 )
                y_test.push_back( std::vector<double>( {row[j]} ) );
            x_test.push_back(temp);
            temp.clear();
        }
    }   
    // end of extracting


    // writing sample of data --------------------------------------------
    /*
    for( std::string col : header_row ) { std::cout << col << " ";}
    std::cout << "\n";
    for(int i = 0; i < 10; i++) {
        for(int j = 0; j < results[0].size(); j++) {
            std::cout << results[i][j] << " ";
        }
        std::cout << "\n";
    }
    */
    // end of writting sample 


    // creating perceptron -----------------------------------------------
    Perceptron perc("sigmoid", "binary-crossentropy");
    perc.train(x_train, y_train, 200, 16, "AdaDelta");
    //Mat y_pred = perc.predict(x_test);

    std::cout << perc << "\n";
    perc.save("./data_files/ClassPerc.4h");

return 0;
}