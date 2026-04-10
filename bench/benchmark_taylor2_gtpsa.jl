using GTPSA
using Printf

function make_dense_series(N::Int, tag::Int)
    d = Descriptor(2, N)
    x, y = vars(d)
    f = TPS(1.0 + 0.1 * tag, use=d)
    for deg in 1:N
        for m in 0:deg
            n = deg - m
            a = 0.01 * (13m + 7n + 3tag + 1)
            b = 0.02 * ((m + 1) * (n + 2) + tag)
            coeff = sin(a) + cos(b)
            f += coeff * (x^m) * (y^n)
        end
    end
    return f
end

function checksum(r, N)
    ax = 0.123
    ay = -0.087
    s = 0.0
    for d in 0:N
        for m in 0:d
            n = d - m
            s += r[[m, n]] * ax^m * ay^n
        end
    end
    return s
end

function time_one(op, f, g, N::Int, repeat::Int, sink_ref)
    t0 = time()
    for _ in 1:repeat
        r = op(f, g)
        sink_ref[] += checksum(r, N)
    end
    return 1000 * (time() - t0) / repeat
end

function main()
    N = 50
    coeffs = (N + 1) * (N + 2) ÷ 2
    repeat = 100

    f = make_dense_series(N, 1)
    g = make_dense_series(N, 2)
    g = g - g[[0]] + 2.0
    ratop(f, g) = (f * f + 2f * g + g * g) / (1 + f - 0.5g)
    sink = Ref(0.0)

    # warmup
    sink[] += checksum(f * g, N)
    sink[] += checksum(f / g, N)
    sink[] += checksum(ratop(f, g), N)

    # measure
    mul_ms = time_one(*, f, g, N, repeat, sink)
    div_ms = time_one(/, f, g, N, repeat, sink)
    rat_ms = time_one(ratop, f, g, N, repeat, sink)

    @printf("N=%d coeffs=%d mul=%.3f ms div=%.3f ms rat=%.3f ms sink=%.15g\n",
        N, coeffs, mul_ms, div_ms, rat_ms, sink[])
end

main()
