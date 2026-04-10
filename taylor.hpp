// ===================================================
// taylor.hpp: Taylor-mode Automatic Differentiation
// Author: Li-Yuan Chiang
// ===================================================

#pragma once

#include <cassert>
#include <cmath>
#include <ostream>
#include <vector>

// ====================================================================
// Taylor 1D
// ====================================================================
// 1. std::vector for coefficients storage
// 2. dynamically sizable, but resize() is never called after
// 3. the functions assume equal size inputs
// ====================================================================
template <typename T> struct Taylor {
    int order;
    std::vector<T> c;
    Taylor(int n) : order(n), c(n + 1, 0) {}
    Taylor(int n, const T &s) : order(n), c(n + 1, 0) { c[0] = s; }
    Taylor &operator=(const T &s) {
        c[0] = s;
        for (int i = 1; i <= order; ++i) {
            c[i] = 0;
        }
        return *this;
    }
};

// variable: [0, 1, 0, 0, ...]
template <typename T> Taylor<T> var(int n) {
    Taylor<T> t(n);
    if (n >= 1) {
        t.c[1] = T{1};
    }
    return t;
}

// printing support
template <typename T> std::ostream &operator<<(std::ostream &os, const Taylor<T> &t) {
    os << "[";
    for (int i = 0; i <= t.order; i++) {
        os << t.c[i] << (i == t.order ? "" : ", ");
    }
    os << "]";
    return os;
}

// Taylor += Taylor
template <typename T> void operator+=(Taylor<T> &t1, const Taylor<T> &t2) {
    assert(t1.order == t2.order);
    for (int i = 0; i <= t1.order; i++) {
        t1.c[i] += t2.c[i];
    }
}

// Taylor + Taylor
template <typename T> Taylor<T> operator+(Taylor<T> t1, const Taylor<T> &t2) {
    t1 += t2;
    return t1;
}

// Taylor += Scalar
template <typename T> void operator+=(Taylor<T> &t1, const T &s) { t1.c[0] += s; }

// Taylor + Scalar
template <typename T> Taylor<T> operator+(Taylor<T> t1, const T &s) {
    t1.c[0] += s;
    return t1;
}

// Scalar + Taylor
template <typename T> Taylor<T> operator+(const T &s, Taylor<T> t1) {
    t1.c[0] += s;
    return t1;
}

// -Taylor
template <typename T> Taylor<T> operator-(Taylor<T> t) {
    for (int i = 0; i <= t.order; i++) {
        t.c[i] = -t.c[i];
    }
    return t;
}

// Taylor -= Taylor
template <typename T> void operator-=(Taylor<T> &t1, const Taylor<T> &t2) {
    assert(t1.order == t2.order);
    for (int i = 0; i <= t1.order; i++) {
        t1.c[i] -= t2.c[i];
    }
}

// Taylor - Taylor
template <typename T> Taylor<T> operator-(Taylor<T> t1, const Taylor<T> &t2) {
    t1 -= t2;
    return t1;
}

// Taylor -= Scalar
template <typename T> void operator-=(Taylor<T> &t1, const T &s) { t1.c[0] -= s; }

// Taylor - Scalar
template <typename T> Taylor<T> operator-(Taylor<T> t1, const T &s) {
    t1.c[0] -= s;
    return t1;
}

// Scalar - Taylor
template <typename T> Taylor<T> operator-(const T &s, Taylor<T> t1) { return -t1 + s; }

// Taylor * Taylor
template <typename T> Taylor<T> operator*(const Taylor<T> &t1, const Taylor<T> &t2) {
    assert(t1.order == t2.order);
    const int n = t1.order;
    Taylor<T> res(n);
    for (int k = 0; k <= n; ++k) {
        T sum = 0;
        for (int i = 0; i <= k; ++i) {
            sum += t1.c[i] * t2.c[k - i];
        }
        res.c[k] = sum;
    }
    return res;
}

// Taylor *= Taylor
template <typename T> void operator*=(Taylor<T> &t1, const Taylor<T> &t2) {
    assert(t1.order == t2.order);
    t1 = t1 * t2;
}

// Taylor *= Scalar
template <typename T> void operator*=(Taylor<T> &t1, const T &s) {
    for (int i = 0; i <= t1.order; i++) {
        t1.c[i] *= s;
    }
}

// Taylor * Scalar
template <typename T> Taylor<T> operator*(Taylor<T> t1, const T &s) {
    for (int i = 0; i <= t1.order; i++) {
        t1.c[i] *= s;
    }
    return t1;
}

// Scalar * Taylor
template <typename T> Taylor<T> operator*(const T &s, Taylor<T> t1) {
    for (int i = 0; i <= t1.order; i++) {
        t1.c[i] *= s;
    }
    return t1;
}

// Taylor / Taylor
template <typename T> Taylor<T> operator/(const Taylor<T> &h, const Taylor<T> &g) {
    assert(h.order == g.order);
    assert(g.c[0] != T{0});
    const int n = h.order;
    Taylor<T> f = h;
    for (int k = 0; k <= n; k++) {
        for (int j = 1; j <= k; j++) {
            f.c[k] -= g.c[j] * f.c[k - j];
        }
        f.c[k] /= g.c[0];
    }
    return f;
}

// Taylor /= Taylor
template <typename T> void operator/=(Taylor<T> &h, const Taylor<T> &g) { h = h / g; }

// Taylor /= Scalar
template <typename T> void operator/=(Taylor<T> &f, const T &s) {
    for (int i = 0; i <= f.order; i++) {
        f.c[i] /= s;
    }
}

// Taylor / Scalar
template <typename T> Taylor<T> operator/(Taylor<T> f, const T &s) {
    for (int i = 0; i <= f.order; i++) {
        f.c[i] /= s;
    }
    return f;
}

// Scalar / Taylor
template <typename T> Taylor<T> operator/(const T &s, Taylor<T> f) {
    assert(f.c[0] != T{0});
    const int n = f.order;
    Taylor<T> S(n, s);
    return S / f;
}

// comparison
template <typename T> bool operator==(const Taylor<T> &a, const Taylor<T> &b) { return a.order == b.order && a.c == b.c; }
template <typename T> bool operator!=(const Taylor<T> &a, const Taylor<T> &b) { return !(a == b); }

// derivative
template <typename T> Taylor<T> deriv(const Taylor<T> &f) {
    Taylor<T> g(f.order);
    for (int i = 1; i <= f.order; ++i) {
        g.c[i - 1] = T(i) * f.c[i];
    }
    g.c[f.order] = T{0};
    return g;
}

// integral
template <typename T> Taylor<T> integral(const Taylor<T> &f) {
    Taylor<T> g(f.order);
    g.c[0] = T{0};
    for (int i = 1; i <= f.order; ++i) {
        g.c[i] = f.c[i - 1] / T(i);
    }
    return g;
}

// inverse
template <typename T> Taylor<T> inv(const Taylor<T> &f) {
    assert(f.c[0] != T{0});
    Taylor<T> g(f.order);
    const T f0inv = T{1} / f.c[0];
    g.c[0] = f0inv;
    for (int i = 1; i <= f.order; ++i) {
        T s = T{0};
        for (int j = 1; j <= i; ++j) {
            s += f.c[j] * g.c[i - j];
        }
        g.c[i] = -s * f0inv;
    }
    return g;
}

// exponential
template <typename T> Taylor<T> exp(const Taylor<T> &f) {
    Taylor<T> y(f.order);
    using std::exp;
    y.c[0] = exp(f.c[0]);
    for (int n = 1; n <= f.order; ++n) {
        T s = T{0};
        for (int k = 1; k <= n; ++k) {
            s += T(k) * f.c[k] * y.c[n - k];
        }
        y.c[n] = s / T(n);
    }
    return y;
}

// logarithm
template <typename T> Taylor<T> log(const Taylor<T> &f) {
    assert(f.c[0] != T{0});
    Taylor<T> y(f.order);
    using std::log;
    y.c[0] = log(f.c[0]);
    const auto yp = deriv(f) * inv(f);
    for (int i = 1; i <= f.order; ++i) {
        y.c[i] = yp.c[i - 1] / T(i);
    }
    return y;
}

// trigonometric functions
template <typename T> inline void sincos(Taylor<T> &s, Taylor<T> &c, const Taylor<T> &f) {
    using std::cos;
    using std::sin;
    s = Taylor<T>(f.order);
    c = Taylor<T>(f.order);
    s.c[0] = sin(f.c[0]);
    c.c[0] = cos(f.c[0]);
    Taylor<T> fp = deriv(f);
    for (int n = 1; n <= f.order; ++n) {
        T ss = T{0};
        T cc = T{0};
        for (int k = 0; k <= n - 1; ++k) {
            ss += c.c[k] * fp.c[n - 1 - k];
            cc += s.c[k] * fp.c[n - 1 - k];
        }
        s.c[n] = ss / T(n);
        c.c[n] = -cc / T(n);
    }
}
template <typename T> Taylor<T> sin(const Taylor<T> &f) {
    Taylor<T> s(f.order), c(f.order);
    sincos(s, c, f);
    return s;
}
template <typename T> Taylor<T> cos(const Taylor<T> &f) {
    Taylor<T> s(f.order), c(f.order);
    sincos(s, c, f);
    return c;
}
template <typename T> Taylor<T> tan(const Taylor<T> &f) {
    Taylor<T> s(f.order), c(f.order);
    sincos(s, c, f);
    return s / c;
}

// integer powers
template <typename T> Taylor<T> powi(Taylor<T> base, int n) {
    assert(n >= 0);
    Taylor<T> result(base.order, T{1});
    while (n > 0) {
        if (n & 1)
            result *= base;
        n >>= 1;
        if (n)
            base *= base;
    }
    return result;
}

// generic powers
template <typename Ta, typename T> Taylor<T> pow(const Taylor<T> &f, const Ta &a_in) {
    const T a = static_cast<T>(a_in);
    const int N = f.order;
    const T f0 = f.c[0];
    assert(f0 != T{0});
    Taylor<T> y(N);
    using std::pow;
    y.c[0] = pow(f0, a);
    Taylor<T> g = deriv(f) * inv(f);
    for (int n = 1; n <= N; ++n) {
        T s = T{0};
        for (int k = 0; k <= n - 1; ++k) {
            s += g.c[k] * y.c[n - 1 - k];
        }
        y.c[n] = (a * s) / T(n);
    }
    return y;
}

// square root
template <typename T> Taylor<T> sqrt(const Taylor<T> &f) {
    using std::sqrt;
    assert(f.c[0] != T{0});
    // assert(f.c[0] > T{0});
    return pow(f, T{0.5});
}

// ====================================================================
// Taylor 2D
// ====================================================================
// Truncated with total order (m + n)
// (m, n) triangle --> getID(m, n) continuous indices
// Order: (0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), ...
// ====================================================================
constexpr int tri(int d) { return d * (d + 1) / 2; }
constexpr int size_for_order(int n) { return (n + 1) * (n + 2) / 2; }
constexpr int getID(int m, int n) { return tri(m + n) + n; }
template <typename T> struct Taylor2 {
    int order;
    std::vector<T> c;
    Taylor2(int n) : order(n), c(size_for_order(n), T{0}) {}
    Taylor2(int n, const T &s) : order(n), c(size_for_order(n), T{0}) { c[0] = s; }
    Taylor2 &operator=(const T &s) {
        c[0] = s;
        for (int i = 1; i < (int)c.size(); ++i)
            c[i] = T{0};
        return *this;
    }
    T &operator()(int m, int n) {
        assert(m >= 0 && n >= 0 && m + n <= order);
        return c[getID(m, n)];
    }
    const T &operator()(int m, int n) const {
        assert(m >= 0 && n >= 0 && m + n <= order);
        return c[getID(m, n)];
    }
    int size() const { return (int)c.size(); }
};

// First variable, i.e. [[0], [1, 0], ...]
template <typename T> Taylor2<T> var1(int n) {
    Taylor2<T> t(n);
    if (n >= 1) {
        t(1, 0) = T{1};
    }
    return t;
}

// Second variable, i.e. [[0], [0, 1], ...]
template <typename T> Taylor2<T> var2(int n) {
    Taylor2<T> t(n);
    if (n >= 1) {
        t(0, 1) = T{1};
    }
    return t;
}

// printing support
template <typename T> std::ostream &operator<<(std::ostream &os, const Taylor2<T> &t) {
    os << "[";
    for (int m = 0; m <= t.order; m++) {
        os << "[";
        for (int n = 0; n <= m; n++) {
            os << t(m - n, n);
            if (n != m) {
                os << ", ";
            }
        }
        os << "]";
        if (m != t.order) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

// Taylor += Taylor
template <typename T> void operator+=(Taylor2<T> &a, const Taylor2<T> &b) {
    assert(a.order == b.order);
    for (int i = 0; i < a.size(); ++i) {
        a.c[i] += b.c[i];
    }
}

// Taylor + Taylor
template <typename T> Taylor2<T> operator+(Taylor2<T> a, const Taylor2<T> &b) {
    a += b;
    return a;
}

// Taylor += Scalar
template <typename T> void operator+=(Taylor2<T> &a, const T &s) { a.c[0] += s; }

// Taylor + Scalar
template <typename T> Taylor2<T> operator+(Taylor2<T> a, const T &s) {
    a.c[0] += s;
    return a;
}

// Scalar + Taylor
template <typename T> Taylor2<T> operator+(const T &s, Taylor2<T> a) {
    a.c[0] += s;
    return a;
}

// -Taylor
template <typename T> Taylor2<T> operator-(Taylor2<T> a) {
    for (auto &x : a.c) {
        x = -x;
    }
    return a;
}

// Taylor -= Taylor
template <typename T> void operator-=(Taylor2<T> &a, const Taylor2<T> &b) {
    assert(a.order == b.order);
    for (int i = 0; i < a.size(); ++i) {
        a.c[i] -= b.c[i];
    }
}

// Taylor - Taylor
template <typename T> Taylor2<T> operator-(Taylor2<T> a, const Taylor2<T> &b) {
    a -= b;
    return a;
}

// Taylor -= Scalar
template <typename T> void operator-=(Taylor2<T> &a, const T &s) { a.c[0] -= s; }

// Taylor - Scalar
template <typename T> Taylor2<T> operator-(Taylor2<T> a, const T &s) {
    a.c[0] -= s;
    return a;
}

// Scalar - Taylor
template <typename T> Taylor2<T> operator-(const T &s, Taylor2<T> a) { return -a + s; }

// Taylor *= Scalar
template <typename T> void operator*=(Taylor2<T> &a, const T &s) {
    for (auto &x : a.c) {
        x *= s;
    }
}

// Taylor * Scalar
template <typename T> Taylor2<T> operator*(Taylor2<T> a, const T &s) {
    a *= s;
    return a;
}

// Scalar * Taylor
template <typename T> Taylor2<T> operator*(const T &s, Taylor2<T> a) {
    a *= s;
    return a;
}

// Taylor /= Scalar
template <typename T> void operator/=(Taylor2<T> &a, const T &s) {
    for (auto &x : a.c) {
        x /= s;
    }
}

// Taylor / Scalar
template <typename T> Taylor2<T> operator/(Taylor2<T> a, const T &s) {
    a /= s;
    return a;
}

// Scalar / Taylor
template <typename T> Taylor2<T> operator/(const T &s, Taylor2<T> f) { return Taylor2<T>(f.order, s) / f; }

// Taylor * Taylor
template <typename T> Taylor2<T> operator*(const Taylor2<T> &a, const Taylor2<T> &b) {
    assert(a.order == b.order);
    const int N = a.order;
    Taylor2<T> r(N);
    for (int d = 0; d <= N; ++d) {
        for (int m = 0; m <= d; ++m) {
            int n = d - m;
            T sum = T{0};
            for (int i = 0; i <= m; ++i) {
                for (int j = 0; j <= n; ++j) {
                    sum += a(i, j) * b(m - i, n - j);
                }
            }
            r(m, n) = sum;
        }
    }
    return r;
}

// Taylor *= Taylor
template <typename T> void operator*=(Taylor2<T> &a, const Taylor2<T> &b) { a = a * b; }

// Taylor / Taylor
template <typename T> Taylor2<T> operator/(const Taylor2<T> &h, const Taylor2<T> &g) {
    assert(h.order == g.order);
    const int N = h.order;
    Taylor2<T> f(N);
    assert(g(0, 0) != T{0});
    for (int d = 0; d <= N; ++d) {
        for (int m = 0; m <= d; ++m) {
            int n = d - m;
            T rhs = h(m, n);
            for (int i = 0; i <= m; ++i) {
                for (int j = 0; j <= n; ++j) {
                    if (i == 0 && j == 0) {
                        continue;
                    }
                    rhs -= g(i, j) * f(m - i, n - j);
                }
            }
            f(m, n) = rhs / g(0, 0);
        }
    }
    return f;
}

// Taylor /= Taylor
template <typename T> void operator/=(Taylor2<T> &a, const Taylor2<T> &b) { a = a / b; }

// Comparison
template <typename T> bool operator==(const Taylor2<T> &a, const Taylor2<T> &b) { return a.order == b.order && a.c == b.c; }
template <typename T> bool operator!=(const Taylor2<T> &a, const Taylor2<T> &b) { return !(a == b); }

// Partial derivatives
template <typename T> Taylor2<T> deriv1(const Taylor2<T> &f) {
    Taylor2<T> g(f.order);
    for (int d = 1; d <= f.order; ++d) {
        for (int m = 1; m <= d; ++m) {
            int n = d - m;
            g(m - 1, n) = T(m) * f(m, n);
        }
    }
    return g;
}
template <typename T> Taylor2<T> deriv2(const Taylor2<T> &f) {
    Taylor2<T> g(f.order);
    for (int d = 1; d <= f.order; ++d) {
        for (int m = 0; m <= d - 1; ++m) {
            int n = d - m;
            g(m, n - 1) = T(n) * f(m, n);
        }
    }
    return g;
}

// Integral against var1() and var2()
template <typename T> Taylor2<T> integral1(const Taylor2<T> &f) {
    Taylor2<T> g(f.order);
    for (int d = 0; d <= f.order - 1; ++d) {
        for (int m = 0; m <= d; ++m) {
            int n = d - m;
            g(m + 1, n) = f(m, n) / T(m + 1);
        }
    }
    return g;
}
template <typename T> Taylor2<T> integral2(const Taylor2<T> &f) {
    Taylor2<T> g(f.order);
    for (int d = 0; d <= f.order - 1; ++d) {
        for (int m = 0; m <= d; ++m) {
            int n = d - m;
            g(m, n + 1) = f(m, n) / T(n + 1);
        }
    }
    return g;
}

// inverse
template <typename T> Taylor2<T> inv(const Taylor2<T> &f) {
    assert(f(0, 0) != T{0});
    const int N = f.order;
    Taylor2<T> g(N);
    const T f00inv = T{1} / f(0, 0);
    g(0, 0) = f00inv;
    for (int d = 1; d <= N; ++d) {
        for (int m = 0; m <= d; ++m) {
            int n = d - m;
            T s = T{0};
            for (int i = 0; i <= m; ++i) {
                for (int j = 0; j <= n; ++j) {
                    if (i == 0 && j == 0)
                        continue;
                    s += f(i, j) * g(m - i, n - j);
                }
            }
            g(m, n) = -s * f00inv;
        }
    }
    return g;
}

// exponential
template <typename T> Taylor2<T> exp(const Taylor2<T> &f) {
    const int N = f.order;
    Taylor2<T> y(N);
    using std::exp;
    y(0, 0) = exp(f(0, 0));
    for (int d = 0; d <= N - 1; ++d) {
        for (int m = 0; m <= d; ++m) {
            int n = d - m;
            T s = T{0};
            for (int i = 0; i <= m; ++i) {
                for (int j = 0; j <= n; ++j) {
                    s += T(i + 1) * f(i + 1, j) * y(m - i, n - j);
                }
            }
            y(m + 1, n) = s / T(m + 1);
        }
    }
    return y;
}

// logarithm
template <typename T> Taylor2<T> log(const Taylor2<T> &f) {
    assert(f(0, 0) != T{0});
    const int N = f.order;
    Taylor2<T> y = integral1(deriv1(f) * inv(f));
    Taylor<T> edge(N);
    for (int n = 0; n <= N; ++n) {
        edge.c[n] = f(0, n);
    }
    Taylor<T> ledge = log(edge);
    for (int n = 0; n <= N; ++n) {
        y(0, n) = ledge.c[n];
    }
    return y;
}

// trigonometric functions
template <typename T> inline void sincos(Taylor2<T> &s, Taylor2<T> &c, const Taylor2<T> &f) {
    using std::cos;
    using std::sin;
    const int N = f.order;
    s = Taylor2<T>(N);
    c = Taylor2<T>(N);
    Taylor<T> edge(N);
    for (int n = 0; n <= N; ++n) {
        edge.c[n] = f(0, n);
    }
    Taylor<T> sedge(N), cedge(N);
    sincos(sedge, cedge, edge);
    for (int n = 0; n <= N; ++n) {
        s(0, n) = sedge.c[n];
        c(0, n) = cedge.c[n];
    }
    Taylor2<T> fp = deriv1(f);
    for (int d = 0; d <= N - 1; ++d) {
        for (int m = 0; m <= d; ++m) {
            int n = d - m;
            T ss = T{0};
            T cc = T{0};
            for (int i = 0; i <= m; ++i) {
                for (int j = 0; j <= n; ++j) {
                    ss += c(i, j) * fp(m - i, n - j);
                    cc += s(i, j) * fp(m - i, n - j);
                }
            }
            s(m + 1, n) = ss / T(m + 1);
            c(m + 1, n) = -cc / T(m + 1);
        }
    }
}
template <typename T> Taylor2<T> sin(const Taylor2<T> &f) {
    Taylor2<T> s(f.order), c(f.order);
    sincos(s, c, f);
    return s;
}
template <typename T> Taylor2<T> cos(const Taylor2<T> &f) {
    Taylor2<T> s(f.order), c(f.order);
    sincos(s, c, f);
    return c;
}
template <typename T> Taylor2<T> tan(const Taylor2<T> &f) {
    Taylor2<T> s(f.order), c(f.order);
    sincos(s, c, f);
    return s / c;
}

// integer powers
template <typename T> Taylor2<T> powi(Taylor2<T> base, int n) {
    assert(n >= 0);
    Taylor2<T> result(base.order, T{1});
    while (n > 0) {
        if (n & 1)
            result *= base;
        n >>= 1;
        if (n)
            base *= base;
    }
    return result;
}

// generic powers
template <typename Ta, typename T> Taylor2<T> pow(const Taylor2<T> &f, const Ta &a_in) {
    const T a = static_cast<T>(a_in);
    const int N = f.order;
    const T f00 = f(0, 0);
    assert(f00 != T{0});
    Taylor2<T> y(N);
    Taylor<T> edge(N);
    for (int n = 0; n <= N; ++n) {
        edge.c[n] = f(0, n);
    }
    Taylor<T> yedge = pow(edge, a);
    for (int n = 0; n <= N; ++n) {
        y(0, n) = yedge.c[n];
    }
    Taylor2<T> g = deriv1(f) * inv(f);
    for (int d = 0; d <= N - 1; ++d) {
        for (int m = 0; m <= d; ++m) {
            int n = d - m;
            T s = T{0};
            for (int i = 0; i <= m; ++i) {
                for (int j = 0; j <= n; ++j) {
                    s += g(i, j) * y(m - i, n - j);
                }
            }
            y(m + 1, n) = (a * s) / T(m + 1);
        }
    }
    return y;
}

// square root
template <typename T> Taylor2<T> sqrt(const Taylor2<T> &f) {
    using std::sqrt;
    assert(f(0, 0) != T{0});
    // assert(f(0, 0) > T{0});
    return pow(f, T{0.5});
}
