CMAKE_MINIMUM_REQUIRED( VERSION 3.0.0 )

set( CMAKE_CXX_STANDARD 14 )

include_directories( include )
set( SOURCES src/main.cpp src/perceptron.cpp )

add_executable( main ${SOURCES} )