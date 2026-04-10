# taylorcpp

_A lightweight, header-only C++ library for dense low-dimensional truncated Taylor algebra._

### Motivation

In my previous research, I often needed to compute truncated Taylor series coefficients efficiently. The amazing Julia library [TaylorSeries.jl](https://github.com/JuliaDiff/TaylorSeries.jl) played an important role in enabling many of these computations. As my research project evolved, I became interested in building a minimal Taylor series library in C++, both to better integrate with my personal workflow and to "reinvent the wheel" as a way to understand the underlying algorithms.

This led to the creation of `taylorcpp`, a header-only library that supports generic numeric types via C++ templates and focuses on clarity and performance for low-dimensional problems. Despite its simplicity and lack of aggressive optimization, the implementation turned out to be surprisingly efficient, which motivated me to share it.

### Main features

`taylorcpp` is a Taylor-mode automatic differentiation (AD) library. It computes Taylor coefficients numerically via operator overloading (`+`, `-`, `*`, `/`) and supports commonly used elementary functions such as `exp`, `log`, `pow`, and trigonometric functions.

The library currently provides two main types:

- `Taylor<T>` (1D expansion)

  Computes coefficients $f_n$ up to order $N$:
  $$
  f(x_0 + \delta x) = \sum_{n=0}^{N} f_n \, \delta x^n + \mathcal{O}(\delta x^{N+1}).
  $$

- `Taylor2<T>` (2D total-degree expansion)

  Computes coefficients $f_{m,n}$ on a dense triangular grid:
  $$
  f(x_0 + \delta x, y_0 + \delta y)
  = \sum_{0 \leq m + n \leq N} f_{m,n} \, \delta x^m \, \delta y^n.
  $$

In both cases, coefficients are stored in a contiguous `std::vector<T>` for memory locality. This choice was intentional for ease of use and flexibility in the initial designation of expansion order, but the library ensures that no dynamic resizing is performed afterward.

### Project scope

`taylorcpp` is not intended to be a full-fledged AD framework but instead a lightweight and efficient tool for specific use cases:
- low-dimensional (1D and 2D) Taylor expansions, 
- dense coefficients storage with total-degree truncation,
- header-only dependencies, and
- generic numeric types.

### Demo
```cpp
// C++17
#include <iostream>
#include <taylor.hpp>

int main() {
    // initialize with order
    int order = 5;
    Taylor<double> t(order); 

    // initialize with a variable
    auto x = var<double>(order);

    // Taylor expand a Planck distribution
    std::cout << 1.0 / (exp(x - 0.42) - 1.0) << std::endl;
    // outputs [-2.91585, -5.58633, -13.4957, -32.1381, -76.5163, -182.181]

    return 0;
}
```

### Comparison with existing libraries

Here, we compare `taylorcpp` with two representative libraries:

- [TaylorSeries.jl](https://github.com/JuliaDiff/TaylorSeries.jl): a feature-rich Julia library for general Taylor series computations 
- [GTPSA](https://github.com/bmad-sim/GTPSA.jl): a truncated power series algebra (TPSA) engine with a highly optimized C backend

All tests are performed on 2D Taylor series with a truncation of $N = 50$ (1326 coefficients). We consider dense 2D series where the coefficients are populated manually with a deterministic rule. The benchmark scripts can be found in `bench/`. The Julia benchmarks are properly warmed up to avoid JIT latency, and all implementations were verified to produce consistent results down to double‑precision accuracy through a weighted sum over all coefficients. (Note: Benchmarks for these libraries were written with LLM assistance to ensure consistent structure and avoid bias toward a specific language. All results were manually verified and audited.)


| Library                    | Mul (ms) | Div (ms) | Rational (ms) |
|----------------------------|----------|----------|---------------|
| `taylorcpp`                | 0.27     | 0.32     | 1.07          |
| `TaylorSeries.jl`          | 1.38     | 1.35     | 5.47          |
| `GTPSA.jl` (Julia wrapper) | 0.15     | 2.80     | 3.20          |

`taylorcpp` significantly outperforms `TaylorSeries.jl` in this setting. Compared with `GTPSA.jl`, `taylorcpp` is competitive overall and can be faster in tasks involving divisions and rational expressions. This shows that a straightforward, specialized implementation can be highly effective in a particular context.

### Future directions

While the current results are encouraging, the library still lacks several features that would be important for broader applications. Extending the implementation to general dimensions is a natural next step and raises nontrivial questions about indexing and memory layout. Implementing a more complete set of functions commonly used in physics and engineering would be helpful. It would also be interesting to explore compile‑time fixed order via `std::array`, as well as support for sparse Taylor series.

`taylorcpp` is under active development and is intended to be part of a larger personal numerical-computation project. Suggestions and feedback are always welcome.
