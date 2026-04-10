#include "../taylor.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>

using Clock = std::chrono::steady_clock;

std::pair<Taylor2<double>, Taylor2<double>> make_dense_series(int N, int tag) {
    auto x = var1<double>(N);
    auto y = var2<double>(N);
    Taylor2<double> f(N, 1.0 + 0.1 * tag);
    std::vector<Taylor2<double>> xpows{Taylor2<double>(N, 1.0)};
    std::vector<Taylor2<double>> ypows{Taylor2<double>(N, 1.0)};
    for (int k = 1; k <= N; ++k) {
        xpows.push_back(xpows.back() * x);
        ypows.push_back(ypows.back() * y);
    }
    for (int d = 1; d <= N; ++d) {
        for (int m = 0; m <= d; ++m) {
            int n = d - m;
            double a = 0.01 * (13.0 * m + 7.0 * n + 3.0 * tag + 1.0);
            double b = 0.02 * ((m + 1.0) * (n + 2.0) + tag);
            double coeff = std::sin(a) + std::cos(b);
            f += coeff * xpows[m] * ypows[n];
        }
    }
    return {f, x};
}

double checksum(const Taylor2<double> &r) {
    constexpr double ax = 0.123;
    constexpr double ay = -0.087;
    double s = 0.0;
    for (int d = 0; d <= r.order; ++d) {
        for (int m = 0; m <= d; ++m) {
            int n = d - m;
            s += r(m, n) * std::pow(ax, m) * std::pow(ay, n);
        }
    }
    return s;
}

template <typename Op> double time_one(Op op, const Taylor2<double> &f, const Taylor2<double> &g, int repeat, double &sink) {
    auto t0 = Clock::now();
    for (int i = 0; i < repeat; ++i) {
        auto r = op(f, g);
        sink += checksum(r);
    }
    auto t1 = Clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / repeat;
}

int main() {
    const int N = 50;
    const int coeffs = (N + 1) * (N + 2) / 2;
    const int repeat = 100;

    auto [f, _x] = make_dense_series(N, 1);
    auto [g_raw, _y] = make_dense_series(N, 2);
    auto g = g_raw - g_raw(0, 0) + 2.0;
    auto rational_op = [](const auto &u, const auto &v) { return (u * u + 2.0 * u * v + v * v) / (1.0 + u - 0.5 * v); };
    double sink = 0.0;

    // warmup
    sink += checksum(f * g);
    sink += checksum(f / g);
    sink += checksum(rational_op(f, g));

    // measure
    double mul_ms = time_one([](const auto &u, const auto &v) { return u * v; }, f, g, repeat, sink);
    double div_ms = time_one([](const auto &u, const auto &v) { return u / v; }, f, g, repeat, sink);
    double rat_ms = time_one(rational_op, f, g, repeat, sink);

    std::printf("N=%d coeffs=%d mul=%.3f ms div=%.3f ms rat=%.3f ms sink=%.15g\n", N, coeffs, mul_ms, div_ms, rat_ms, sink);
    return 0;
}
