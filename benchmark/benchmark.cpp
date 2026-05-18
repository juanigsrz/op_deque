// op_deque benchmark + amortized-cost stress tool.
//
// For each workload, we run the same sequence of operations against
//   (a) op_deque<long long, OpMin>     -- the device under test
//   (b) std::deque<long long>          -- the brute reference
// XOR-ing each get() result. A small-N "oracle" pass demands the two
// checksums match for every workload before timing starts. The timing
// pass repeats the workload at geometrically-spaced sizes N, takes the
// median wall time per trial, and fits log(t) = a*log(N) + b. For an
// amortized-O(1)-per-op data structure the empirical exponent a should
// land near 1.0 with small standard error.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "op_deque.cpp"

using std::deque;
using std::map;
using std::string;
using std::vector;
using std::cerr;
using std::cout;
using std::setw;
using std::left;
using std::right;
using std::fixed;
using std::setprecision;
using std::chrono::steady_clock;
using std::chrono::nanoseconds;
using std::chrono::duration_cast;

template<class T>
struct OpMin {
    T operator()(T& a, T& b) { return a < b ? a : b; }
};

using OD = op_deque<long long, OpMin<long long>>;

// SplitMix64 — non-monotone hash so running min depends on deque contents.
static inline long long h(long long i) {
    uint64_t x = (uint64_t)i + 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return (long long)x;
}

static inline long long brute_min(const deque<long long>& dq) {
    long long m = dq.front();
    for (auto x : dq) if (x < m) m = x;
    return m;
}

// ---------------------------------------------------------------------------
// Workloads. Each pair (run_X, brute_X) must compute identical checksums.
// All workloads perform Theta(N) underlying op_deque calls (with a constant
// op-multiplier that does not depend on N), so log(t) vs log(N) regression
// is meaningful and slope ~ 1.0 confirms amortized O(1) per op.
// ---------------------------------------------------------------------------

// push_back only, get() after each push.
static long long run_push_back(long long N) {
    OD D; long long c = 0;
    for (long long i = 0; i < N; i++) { D.push_back(h(i)); c ^= D.get(); }
    return c;
}
static long long brute_push_back(long long N) {
    deque<long long> D; long long c = 0;
    for (long long i = 0; i < N; i++) { D.push_back(h(i)); c ^= brute_min(D); }
    return c;
}

// push_front only.
static long long run_push_front(long long N) {
    OD D; long long c = 0;
    for (long long i = 0; i < N; i++) { D.push_front(h(i)); c ^= D.get(); }
    return c;
}
static long long brute_push_front(long long N) {
    deque<long long> D; long long c = 0;
    for (long long i = 0; i < N; i++) { D.push_front(h(i)); c ^= brute_min(D); }
    return c;
}

// Alternate push_back / push_front -- keeps the two internal stacks balanced.
static long long run_push_alternate(long long N) {
    OD D; long long c = 0;
    for (long long i = 0; i < N; i++) {
        if (i & 1) D.push_back(h(i)); else D.push_front(h(i));
        c ^= D.get();
    }
    return c;
}
static long long brute_push_alternate(long long N) {
    deque<long long> D; long long c = 0;
    for (long long i = 0; i < N; i++) {
        if (i & 1) D.push_back(h(i)); else D.push_front(h(i));
        c ^= brute_min(D);
    }
    return c;
}

// rotate: pre-fill ~sqrt(N), then push_back/pop_front cycle -- empties s1 and
// retriggers balance() every Theta(sqrt(N)) iters.
static long long run_rotate(long long N) {
    OD D; long long c = 0;
    long long K = std::max((long long)16, (long long)std::sqrt((double)N));
    for (long long i = 0; i < K; i++) D.push_back(h(i));
    for (long long i = 0; i < N; i++) {
        D.push_back(h(K + i));
        D.pop_front();
        c ^= D.get();
    }
    return c;
}
static long long brute_rotate(long long N) {
    deque<long long> D; long long c = 0;
    long long K = std::max((long long)16, (long long)std::sqrt((double)N));
    for (long long i = 0; i < K; i++) D.push_back(h(i));
    for (long long i = 0; i < N; i++) {
        D.push_back(h(K + i));
        D.pop_front();
        c ^= brute_min(D);
    }
    return c;
}

// fill-then-drain from same end.
static long long run_fill_drain_back(long long N) {
    OD D; long long c = 0;
    for (long long i = 0; i < N; i++) { D.push_back(h(i)); c ^= D.get(); }
    for (long long i = 0; i < N; i++) { c ^= D.get(); D.pop_back(); }
    return c;
}
static long long brute_fill_drain_back(long long N) {
    deque<long long> D; long long c = 0;
    for (long long i = 0; i < N; i++) { D.push_back(h(i)); c ^= brute_min(D); }
    for (long long i = 0; i < N; i++) { c ^= brute_min(D); D.pop_back(); }
    return c;
}

// fill-then-drain from opposite end -- forces a long chain of balance() calls
// as s2 empties through s1.
static long long run_fill_drain_front(long long N) {
    OD D; long long c = 0;
    for (long long i = 0; i < N; i++) { D.push_back(h(i)); c ^= D.get(); }
    for (long long i = 0; i < N; i++) { c ^= D.get(); D.pop_front(); }
    return c;
}
static long long brute_fill_drain_front(long long N) {
    deque<long long> D; long long c = 0;
    for (long long i = 0; i < N; i++) { D.push_back(h(i)); c ^= brute_min(D); }
    for (long long i = 0; i < N; i++) { c ^= brute_min(D); D.pop_front(); }
    return c;
}

// adversarial threshold-tickler: pre-fill, then repeatedly push 3, pop_front 1
// to drive the size ratio toward the 3:1 trigger as fast as possible.
static long long run_adversarial(long long N) {
    OD D; long long c = 0;
    long long K = std::max((long long)16, N / 4);
    for (long long i = 0; i < K; i++) {
        if (i & 1) D.push_back(h(i)); else D.push_front(h(i));
    }
    long long cycles = N / 4;
    for (long long k = 0; k < cycles; k++) {
        D.push_back(h(K + 4*k + 0));
        D.push_back(h(K + 4*k + 1));
        D.push_back(h(K + 4*k + 2));
        D.pop_front();
        c ^= D.get();
    }
    return c;
}
static long long brute_adversarial(long long N) {
    deque<long long> D; long long c = 0;
    long long K = std::max((long long)16, N / 4);
    for (long long i = 0; i < K; i++) {
        if (i & 1) D.push_back(h(i)); else D.push_front(h(i));
    }
    long long cycles = N / 4;
    for (long long k = 0; k < cycles; k++) {
        D.push_back(h(K + 4*k + 0));
        D.push_back(h(K + 4*k + 1));
        D.push_back(h(K + 4*k + 2));
        D.pop_front();
        c ^= brute_min(D);
    }
    return c;
}

// sliding window of size sqrt(N) over a stream of length N -- the canonical
// "monotonic deque" use case (e.g. range-min over sliding window).
static long long run_sliding_window(long long N) {
    OD D; long long c = 0;
    long long W = std::max((long long)4, (long long)std::sqrt((double)N));
    if (W > N) W = N;
    for (long long i = 0; i < W; i++) D.push_back(h(i));
    c ^= D.get();
    for (long long i = W; i < N; i++) {
        D.push_back(h(i));
        D.pop_front();
        c ^= D.get();
    }
    return c;
}
static long long brute_sliding_window(long long N) {
    deque<long long> D; long long c = 0;
    long long W = std::max((long long)4, (long long)std::sqrt((double)N));
    if (W > N) W = N;
    for (long long i = 0; i < W; i++) D.push_back(h(i));
    c ^= brute_min(D);
    for (long long i = W; i < N; i++) {
        D.push_back(h(i));
        D.pop_front();
        c ^= brute_min(D);
    }
    return c;
}

// random mix at steady-state size ~N/2 (pre-filled), N random ops.
// Both run and brute MUST consume the same PRNG stream and make the same
// branch choices so checksums match.
static long long run_random(long long N) {
    OD D; long long c = 0;
    long long K = N / 2;
    for (long long i = 0; i < K; i++) {
        if (i & 1) D.push_back(h(i)); else D.push_front(h(i));
    }
    uint64_t s = 0xC0FFEE0123456789ULL;
    for (long long i = 0; i < N; i++) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        int op = (s >> 17) & 0x7;
        long long val = h(K + i);
        switch (op) {
            case 0: case 1: case 2: D.push_back(val); break;
            case 3: case 4: case 5: D.push_front(val); break;
            case 6: if (D.size() > 1) D.pop_back();  break;
            case 7: if (D.size() > 1) D.pop_front(); break;
        }
        if (D.size() > 0) c ^= D.get();
    }
    return c;
}
static long long brute_random(long long N) {
    deque<long long> D; long long c = 0;
    long long K = N / 2;
    for (long long i = 0; i < K; i++) {
        if (i & 1) D.push_back(h(i)); else D.push_front(h(i));
    }
    uint64_t s = 0xC0FFEE0123456789ULL;
    for (long long i = 0; i < N; i++) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        int op = (s >> 17) & 0x7;
        long long val = h(K + i);
        switch (op) {
            case 0: case 1: case 2: D.push_back(val); break;
            case 3: case 4: case 5: D.push_front(val); break;
            case 6: if (D.size() > 1) D.pop_back();  break;
            case 7: if (D.size() > 1) D.pop_front(); break;
        }
        if (!D.empty()) c ^= brute_min(D);
    }
    return c;
}

// ---------------------------------------------------------------------------
struct Workload {
    const char* name;
    long long (*run)(long long);
    long long (*brute)(long long);
};

static const vector<Workload> WORKLOADS = {
    {"push_back",        run_push_back,        brute_push_back},
    {"push_front",       run_push_front,       brute_push_front},
    {"push_alternate",   run_push_alternate,   brute_push_alternate},
    {"rotate",           run_rotate,           brute_rotate},
    {"fill_drain_back",  run_fill_drain_back,  brute_fill_drain_back},
    {"fill_drain_front", run_fill_drain_front, brute_fill_drain_front},
    {"adversarial",      run_adversarial,      brute_adversarial},
    {"sliding_window",   run_sliding_window,   brute_sliding_window},
    {"random",           run_random,           brute_random},
};

static vector<long long> parse_sizes(const string& s) {
    vector<long long> out; string cur;
    for (char c : s) {
        if (c == ',') { if (!cur.empty()) { out.push_back(std::stoll(cur)); cur.clear(); } }
        else cur += c;
    }
    if (!cur.empty()) out.push_back(std::stoll(cur));
    return out;
}

static void print_help(const char* prog) {
    cerr << "Usage: " << prog << " [options]\n"
         << "  --sizes N1,N2,...   Geometric ladder of problem sizes (default: 1e4..1e7).\n"
         << "  --trials K          Trials per (workload, N); median is reported (default: 5).\n"
         << "  --no-oracle         Skip the correctness oracle (default: enabled).\n"
         << "  --oracle-n N        N for the oracle correctness pass (default: 2000).\n"
         << "  --workload NAME     Restrict to a single workload by name.\n"
         << "  --csv-only          Suppress the human summary; print only CSV.\n"
         << "  --help              Show this message.\n"
         << "\nCSV columns: workload,N,trial,time_ns,time_per_op_ns,check\n"
         << "Summary (stderr) reports the slope of log(t) vs log(N) per workload;\n"
         << "for amortized O(1) per op, slope should be ~ 1.0.\n";
}

int main(int argc, char** argv) {
    vector<long long> sizes = {10000, 31623, 100000, 316228, 1000000, 3162278, 10000000};
    int trials = 5;
    bool run_oracle = true;
    long long oracle_n = 2000;
    string only_workload;
    bool csv_only = false;

    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a == "--sizes" && i+1 < argc)       sizes = parse_sizes(argv[++i]);
        else if (a == "--trials" && i+1 < argc) trials = std::atoi(argv[++i]);
        else if (a == "--no-oracle")            run_oracle = false;
        else if (a == "--oracle-n" && i+1 < argc) oracle_n = std::atoll(argv[++i]);
        else if (a == "--workload" && i+1 < argc) only_workload = argv[++i];
        else if (a == "--csv-only")             csv_only = true;
        else if (a == "--help" || a == "-h")    { print_help(argv[0]); return 0; }
        else { cerr << "unknown option: " << a << "\n"; print_help(argv[0]); return 2; }
    }

    if (!csv_only) {
        cerr << "# op_deque benchmark\n"
             << "# sizes:";
        for (auto N : sizes) cerr << " " << N;
        cerr << "\n# trials: " << trials << "\n";
    }

    if (run_oracle) {
        if (!csv_only) cerr << "# Oracle: comparing op_deque get() vs brute std::deque at N=" << oracle_n << "\n";
        bool all_pass = true;
        for (auto& w : WORKLOADS) {
            if (!only_workload.empty() && only_workload != w.name) continue;
            long long a = w.run(oracle_n);
            long long b = w.brute(oracle_n);
            bool pass = (a == b);
            if (!csv_only) {
                cerr << "  " << left << setw(20) << w.name << " " << (pass ? "PASS" : "FAIL");
                if (!pass) cerr << "  (run=" << a << " brute=" << b << ")";
                cerr << "\n";
            }
            if (!pass) all_pass = false;
        }
        if (!all_pass) {
            cerr << "# Oracle FAILED. Aborting before timing.\n";
            return 1;
        }
        if (!csv_only) cerr << "# Oracle PASSED for all selected workloads.\n\n";
    }

    cout << "workload,N,trial,time_ns,time_per_op_ns,check\n";

    struct Row { long long N; vector<long long> ts; };
    map<string, vector<Row>> agg;

    for (auto& w : WORKLOADS) {
        if (!only_workload.empty() && only_workload != w.name) continue;
        for (long long N : sizes) {
            vector<long long> ts;
            for (int t = 0; t < trials; t++) {
                auto t0 = steady_clock::now();
                long long check = w.run(N);
                auto t1 = steady_clock::now();
                long long ns = duration_cast<nanoseconds>(t1 - t0).count();
                cout << w.name << "," << N << "," << t << "," << ns << ","
                     << fixed << setprecision(3) << ((double)ns / (double)N) << "," << check << "\n";
                cout.flush();
                ts.push_back(ns);
            }
            agg[w.name].push_back({N, std::move(ts)});
        }
    }

    if (csv_only) return 0;

    cerr << "\n# Regression of median wall-time vs N (per workload)\n";
    cerr << "# Model:   log(t) = a*log(N) + b      -- amortized O(1)/op => a ~ 1.0\n";
    cerr << "# columns: workload | slope a | std-err | intercept b | t/op @ minN | t/op @ maxN | drift\n";
    cerr << "# (drift = (t/op at max N) / (t/op at min N); ~1.0 confirms flat per-op cost)\n";

    for (auto& [name, rows] : agg) {
        int n = (int)rows.size();
        if (n < 2) {
            cerr << "  " << left << setw(20) << name << " (insufficient data points)\n";
            continue;
        }
        vector<double> xs(n), ys(n);
        long long min_N = rows.front().N, max_N = rows.back().N;
        double t_min_per = 0, t_max_per = 0;
        for (int i = 0; i < n; i++) {
            auto ts = rows[i].ts;
            std::sort(ts.begin(), ts.end());
            long long median = ts[ts.size()/2];
            xs[i] = std::log((double)rows[i].N);
            ys[i] = std::log((double)median);
            double per = (double)median / (double)rows[i].N;
            if (rows[i].N == min_N) t_min_per = per;
            if (rows[i].N == max_N) t_max_per = per;
        }
        double mx = 0, my = 0;
        for (int i = 0; i < n; i++) { mx += xs[i]; my += ys[i]; }
        mx /= n; my /= n;
        double sxx = 0, sxy = 0;
        for (int i = 0; i < n; i++) {
            sxx += (xs[i]-mx)*(xs[i]-mx);
            sxy += (xs[i]-mx)*(ys[i]-my);
        }
        double slope = (sxx > 0) ? sxy / sxx : 0.0;
        double intercept = my - slope * mx;
        double ssr = 0;
        for (int i = 0; i < n; i++) {
            double pred = slope * xs[i] + intercept;
            double r = ys[i] - pred;
            ssr += r*r;
        }
        double slope_se = (n > 2 && sxx > 0) ? std::sqrt(ssr / (n - 2) / sxx) : 0.0;

        cerr << "  " << left << setw(18) << name
             << "  a=" << fixed << setprecision(3) << right << setw(6) << slope
             << " +/- " << setw(5) << slope_se
             << "   b=" << setw(7) << intercept
             << "   " << setprecision(2) << right << setw(8) << t_min_per << " ns/op"
             << "   " << setw(8) << t_max_per << " ns/op"
             << "   " << setprecision(2) << (t_min_per > 0 ? t_max_per / t_min_per : 0.0) << "x\n";
    }
    return 0;
}
