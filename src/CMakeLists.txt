cmake_minimum_required(VERSION 2.8.12)
project(sampling)
set (CMAKE_CXX_STANDARD 11)

find_package(pybind11 REQUIRED)

pybind11_add_module(sampling sampling.cc)
