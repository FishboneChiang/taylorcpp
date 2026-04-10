g++ -O3 -std=c++17 -march=native -DNDEBUG benchmark_taylor2.cpp && ./a.out
julia benchmark_taylor2.jl
julia benchmark_taylor2_gtpsa.jl
