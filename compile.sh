#!/bin/bash

export CC=/usr/bin/clang
export CXX=/usr/bin/clang++

cmake -S . -B build/ -D CMAKE_BUILD_TYPE=Release \
    && cmake --build build \
    && cpplint $(find include src -name "*.hpp" -o -name "*.cpp" -o -name "*.h")
