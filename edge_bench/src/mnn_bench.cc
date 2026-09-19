// MNN edge inference benchmark.
//
// Supports FP32/FP16/INT8 .mnn models on CPU / OpenCL / CUDA / Metal.
//
// The dataset is passed as a directory with the same structure as
// data/datasets/seeds_rgb:
//   <dataset-dir>/images/<name>.png
//   <dataset-dir>/images_with_bg/<name>.png
//   <dataset-dir>/cls_<split>_<tag>_rgb.json
//
// Images are loaded with MNN::CV::imread and preprocessed with
// MNN::CV::ImageProcess (resize + RGB normalization).
//
// Usage:
//   mnn_bench --model model.mnn --backend CPU \
//             --dataset-json data/datasets/OpenSeed-LZU/cls_656_30_rgb_st_crop_test.json \
//             --base-dirs rgb,rgb_with_bg \
//             --max-images 1000 --output-json edge_bench/output/result.json
//
// With --save-pred, the per-image ground-truth and predicted labels are additionally
// stored in the output JSON as "y_true" / "y_pred", laid out base-dir major (all images
// of base_dirs[0], then base_dirs[1], ...). They are omitted by default because the two
// lists dominate the file size (~1 MB per 100k images per model).
#ifndef _MNN_BENCH_H_
#define _MNN_BENCH_H_

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <io.h>
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_types.h>
#endif
#endif
#include <sys/stat.h>

#include <MNN/ErrorCode.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/ImageProcess.hpp>
#include <MNN/MNNForwardType.h>
#include <MNN/Tensor.hpp>

#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Module.hpp>

#include <cv/cv.hpp>
#include <cv/imgcodecs.hpp>

#include <nlohmann/json.hpp>

#include <indicators/progress_bar.hpp>

namespace {

struct Options {
    std::string model_path;
    int backend = MNN_FORWARD_CPU;
    std::string backend_name = "CPU";
    std::string dataset_json;
    std::string image_dir;
    int topk = 5;
    std::string base_dirs = "rgb,rgb_with_bg";
    int max_images = 0;
    int seed = 42;
    std::string output_json;
    int threads = 1;
    int warmup = 10;
    int repeat = 50;
    int width = 224;
    int height = 224;
    int channels = 3;
    // Draw a progress bar on stderr while the benchmark loops run.
    bool show_progress = true;
    // Also write the raw y_true / y_pred lists into --output-json.
    bool save_pred = false;
    // MNN backend precision hint. Precision_Low means "compute in fp16", which
    // returns all-NaN logits on the CUDA backend for every attention architecture we
    // ship (ViT / Swin / MobileViT / MobileNetV4-hybrid). Precision_Normal is fp32 and
    // costs only ~2% more latency, Precision_High is ~5x slower; see --precision help.
    std::string precision = "normal";
};

struct Metrics {
    double latency_ms_mean = 0.0;
    double latency_p50 = 0.0;
    double latency_p90 = 0.0;
    double latency_p99 = 0.0;
    double preprocess_ms_mean = 0.0;
    double preprocess_p50 = 0.0;
    double preprocess_p90 = 0.0;
    double preprocess_p99 = 0.0;
    double inference_ms_mean = 0.0;
    double inference_p50 = 0.0;
    double inference_p90 = 0.0;
    double inference_p99 = 0.0;
    double postprocess_ms_mean = 0.0;
    double postprocess_p50 = 0.0;
    double postprocess_p90 = 0.0;
    double postprocess_p99 = 0.0;
    double throughput = 0.0;
    long max_rss_kb = 0;           // model-level peak sampled RSS in KB
    long mean_rss_kb = 0;          // mean sampled RSS in KB
    long process_peak_rss_kb = 0;  // process-wide peak RSS in KB (high-water mark)
    size_t file_phys_size = 0; // model file size in bytes
    int top1 = 0;
    int top3 = 0;
    int total = 0;
    std::vector<int> y_true;  // ground truth, only filled when a dataset JSON is given
    std::vector<int> y_pred;  // top-1 prediction, parallel to y_true
};

void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s --model <mnn>\n"
        "            [--backend CPU|OPENCL|CUDA|METAL]\n"
        "            [--dataset-json <json>] | [--image-dir <dir>]\n"
        "            [--base-dirs rgb,rgb_with_bg] [--max-images N] [--seed SEED]\n"
        "            [--topk N] [--threads N] [--warmup N] [--repeat N]\n"
        "            [--width N] [--height N] [--precision low|normal|high]\n"
        "            [--no-progress] [--save-pred]\n"
        "            [--output-json <path>]\n"
        "\n"
        "  --precision  MNN backend precision hint (default: normal).\n"
        "               low    = fp16 compute.  On the CUDA backend this returns all-NaN\n"
        "                        logits for attention models (ViT/Swin/MobileViT/\n"
        "                        MobileNetV4-hybrid), which then report one constant\n"
        "                        class and ~0.5%% accuracy.  Do not use it on CUDA.\n"
        "               normal = fp32 compute, correct and ~2%% slower than low\n"
        "               high   = fp32 compute, most conservative and ~5x slower\n"
        "\n"
        "  --no-progress  Do not draw the progress bar (it is also skipped automatically\n"
        "               when stderr is not a terminal).\n"
        "  --save-pred  Write the per-image y_true / y_pred lists into --output-json\n"
        "               (requires --dataset-json). Off by default: the two lists are\n"
        "               about 1 MB per 100k images per model.\n",
        prog);
}

std::string parse_arg(int& i, int argc, char** argv, const std::string& name) {
    if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + name);
    }
    return argv[++i];
}

void parse_args(int argc, char** argv, Options& opt) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (a == "--model") {
            opt.model_path = parse_arg(i, argc, argv, a);
        } else if (a == "--backend") {
            std::string b = parse_arg(i, argc, argv, a);
            if (b == "CPU") { opt.backend = MNN_FORWARD_CPU; opt.backend_name = "CPU"; }
            else if (b == "OPENCL") { opt.backend = MNN_FORWARD_OPENCL; opt.backend_name = "OPENCL"; }
            else if (b == "CUDA") { opt.backend = MNN_FORWARD_CUDA; opt.backend_name = "CUDA"; }
            else if (b == "METAL") { opt.backend = MNN_FORWARD_METAL; opt.backend_name = "METAL"; }
            else { throw std::runtime_error("Unknown backend: " + b); }
        } else if (a == "--dataset-json") {
            opt.dataset_json = parse_arg(i, argc, argv, a);
        } else if (a == "--image-dir") {
            opt.image_dir = parse_arg(i, argc, argv, a);
        } else if (a == "--topk") {
            opt.topk = std::stoi(parse_arg(i, argc, argv, a));
        } else if (a == "--base-dirs") {
            opt.base_dirs = parse_arg(i, argc, argv, a);
        } else if (a == "--max-images") {
            opt.max_images = std::stoi(parse_arg(i, argc, argv, a));
        } else if (a == "--seed") {
            opt.seed = std::stoi(parse_arg(i, argc, argv, a));
        } else if (a == "--output-json") {
            opt.output_json = parse_arg(i, argc, argv, a);
        } else if (a == "--threads") {
            opt.threads = std::stoi(parse_arg(i, argc, argv, a));
        } else if (a == "--warmup") {
            opt.warmup = std::stoi(parse_arg(i, argc, argv, a));
        } else if (a == "--repeat") {
            opt.repeat = std::stoi(parse_arg(i, argc, argv, a));
        } else if (a == "--width") {
            opt.width = std::stoi(parse_arg(i, argc, argv, a));
        } else if (a == "--height") {
            opt.height = std::stoi(parse_arg(i, argc, argv, a));
        } else if (a == "--no-progress") {
            opt.show_progress = false;
        } else if (a == "--save-pred") {
            opt.save_pred = true;
        } else if (a == "--precision") {
            std::string v = parse_arg(i, argc, argv, a);
            if (v != "low" && v != "normal" && v != "high") {
                throw std::runtime_error("--precision must be low, normal or high, got: " + v);
            }
            opt.precision = v;
        } else {
            throw std::runtime_error("Unknown option: " + a);
        }
    }
    if (opt.model_path.empty()) {
        throw std::runtime_error("--model is required");
    }
    if (!opt.dataset_json.empty() && !opt.image_dir.empty()) {
        throw std::runtime_error("--dataset-json and --image-dir are mutually exclusive");
    }
}

// ---------------------------------------------------------------------------
// Dataset loading (uses nlohmann/json)
// ---------------------------------------------------------------------------

struct Sample {
    std::string image_name;
    int label = 0;
};

std::vector<Sample> load_samples(const Options& opt) {
    std::string json_path = opt.dataset_json;
    std::ifstream f(json_path);
    if (!f) {
        throw std::runtime_error("Cannot open file: " + json_path);
    }
    nlohmann::json j;
    f >> j;
    if (!j.contains("images") || !j.contains("labels")) {
        throw std::runtime_error("Missing images/labels in JSON: " + json_path);
    }
    const auto& images = j["images"].get<std::vector<std::string>>();
    const auto& labels = j["labels"].get<std::vector<int>>();
    if (images.size() != labels.size()) {
        throw std::runtime_error("Parse error: images/labels length mismatch");
    }
    std::vector<Sample> samples;
    samples.reserve(images.size());
    for (size_t i = 0; i < images.size(); ++i) {
        samples.push_back({images[i], labels[i]});
    }
    // Optional random subset
    if (opt.max_images > 0 && opt.max_images < static_cast<int>(samples.size())) {
        std::mt19937 rng(opt.seed);
        std::shuffle(samples.begin(), samples.end(), rng);
        samples.resize(opt.max_images);
    }
    return samples;
}

// ---------------------------------------------------------------------------
// Image-directory debugging mode
// ---------------------------------------------------------------------------

std::vector<Sample> load_image_samples(const Options& opt) {
    namespace fs = std::filesystem;
    std::vector<Sample> samples;
    for (const auto& entry : fs::directory_iterator(opt.image_dir)) {
        if (!entry.is_regular_file()) continue;
        auto name = entry.path().filename().string();
        if (name.size() < 4 || name.substr(name.size() - 4) != ".png") continue;
        auto underscore = name.find('_');
        if (underscore == std::string::npos) continue;
        int label = 0;
        try {
            label = std::stoi(name.substr(0, underscore));
        } catch (...) {
            continue;
        }
        samples.push_back({name, label});
    }
    std::sort(samples.begin(), samples.end(),
        [](const Sample& a, const Sample& b) { return a.image_name < b.image_name; });
    // Optional random subset (same convention as dataset mode).
    if (opt.max_images > 0 && opt.max_images < static_cast<int>(samples.size())) {
        std::mt19937 rng(opt.seed);
        std::shuffle(samples.begin(), samples.end(), rng);
        samples.resize(opt.max_images);
    }
    return samples;
}

std::string image_latin_name(const Sample& sample) {
    // classID_latinName_number.png -> latinName (middle segments joined by '_')
    auto name = sample.image_name;
    auto first = name.find('_');
    if (first == std::string::npos) return sample.image_name;
    auto last = name.rfind('_');
    if (last == std::string::npos || last == first) return name;
    return name.substr(first + 1, last - first - 1);
}

// ---------------------------------------------------------------------------

long get_current_rss_kb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return static_cast<long>(pmc.WorkingSetSize / 1024);
    }
    return 0;
#elif defined(__APPLE__)
    mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) == KERN_SUCCESS) {
        return static_cast<long>(info.resident_size / 1024);
    }
    return 0;
#else
    // Linux / Android: read resident set size from /proc/self/statm.
    std::ifstream f("/proc/self/statm");
    std::string total;
    long resident = 0;
    if (f >> total >> resident) {
        long page_size = ::sysconf(_SC_PAGESIZE);
        if (page_size > 0) {
            return static_cast<long>(resident * page_size / 1024);
        }
    }
    return 0;
#endif
}

long get_peak_rss_kb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return static_cast<long>(pmc.PeakWorkingSetSize / 1024);
    }
    return 0;
#else
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
#ifdef __APPLE__
    return static_cast<long>(ru.ru_maxrss / 1024);
#else
    return static_cast<long>(ru.ru_maxrss);
#endif
#endif
}

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double idx = p * (v.size() - 1);
    size_t lo = static_cast<size_t>(idx);
    size_t hi = std::min(lo + 1, v.size() - 1);
    double frac = idx - lo;
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> parts;
    size_t start = 0;
    while (true) {
        size_t pos = s.find(',', start);
        if (pos == std::string::npos) {
            parts.push_back(s.substr(start));
            break;
        }
        parts.push_back(s.substr(start, pos - start));
        start = pos + 1;
    }
    return parts;
}

std::vector<int> top_k(const std::vector<float>& logits, int k) {
    int kk = std::min(k, static_cast<int>(logits.size()));
    std::vector<int> idx(logits.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
        [&](int a, int b) { return logits[a] > logits[b]; });
    return std::vector<int>(idx.begin(), idx.begin() + kk);
}

MNN::Express::VARP resize_and_normalize_hwc(MNN::Express::VARP img, const Options& opt) {
    const std::vector<float> mean = {
        0.34865115f * 255.0f, 0.29936219f * 255.0f, 0.25752143f * 255.0f};
    const std::vector<float> norm = {
        1.0f / (0.2302609f * 255.0f),
        1.0f / (0.19996711f * 255.0f),
        1.0f / (0.17874425f * 255.0f)};

    // Resize + normalize using MNN::CV::resize (mean/norm are in uint8 scale).
    // Returns [H,W,C] float (NHWC), to be converted to NCHW by the caller.
    img = MNN::CV::resize(
        img, MNN::CV::Size(opt.width, opt.height),
        0, 0, MNN::CV::INTER_LINEAR, MNN::CV::COLOR_BGR2RGB, mean, norm);
    return img;
}

MNN::Express::VARP make_synthetic_input(const Options& opt) {
    // Create a random raw image (H, W, C) in the same uint8-like [0,255] range
    // that MNN::CV::imread would produce, then run it through the standard
    // resize and normalization pipeline.
    std::mt19937 rng(opt.seed);
    std::uniform_real_distribution<float> dist(0.0f, 255.0f);
    std::vector<float> data(opt.height * opt.width * opt.channels);
    for (auto& v : data) v = dist(rng);

    MNN::Express::VARP img = MNN::Express::_Const(
        data.data(), {opt.height, opt.width, opt.channels}, MNN::Express::NHWC);
    return img;
}

MNN::Express::VARP load_image(const std::string& img_path, const Options& opt) {
    MNN::Express::VARP img = MNN::CV::imread(img_path, MNN::CV::IMREAD_COLOR);
    if (img == nullptr) {
        throw std::runtime_error("Cannot read image: " + img_path);
    }
    auto* info = img->getInfo();
    if (info == nullptr || info->dim.size() < 3) {
        throw std::runtime_error("Invalid image info: " + img_path);
    }
    return img;
}

void preprocess_to_host(const Options& opt, const Sample& sample, const std::string& base_dir,
                        std::vector<float>& out_nchw) {
    MNN::Express::VARP raw;
    if (opt.dataset_json.empty() && opt.image_dir.empty()) {
        raw = make_synthetic_input(opt);
    } else {
        const std::string dataset_dir = std::filesystem::path(opt.dataset_json).parent_path().string();
        const std::string img_path = (!opt.image_dir.empty())
            ? (opt.image_dir + "/" + sample.image_name)
            : (dataset_dir + "/" + base_dir + "/" + sample.image_name);
        raw = load_image(img_path, opt);
    }
    MNN::Express::VARP hwc = resize_and_normalize_hwc(raw, opt);
    auto* ptr = hwc->readMap<float>();
    if (ptr == nullptr) {
        throw std::runtime_error("Failed to read preprocessed image data");
    }
    out_nchw.assign(opt.channels * opt.height * opt.width, 0.0f);
    for (int h = 0; h < opt.height; ++h) {
        for (int w = 0; w < opt.width; ++w) {
            for (int c = 0; c < opt.channels; ++c) {
                out_nchw[c * opt.height * opt.width + h * opt.width + w] =
                    ptr[(h * opt.width + w) * opt.channels + c];
            }
        }
    }
}

bool stderr_is_tty() {
#ifdef _WIN32
    return _isatty(_fileno(stderr)) != 0;
#else
    return ::isatty(::fileno(stderr)) != 0;
#endif
}

// Progress bar for the benchmark loops, backed by the vendored indicators library.
//
// It writes to stderr so stdout keeps only the machine-readable summary, throttles
// repaints (a terminal write per image would otherwise dominate the runtime of the
// fast models) and stays silent when stderr is not a terminal, so redirected logs and
// the subprocess capture in scripts/run_benchmarks.py stay free of escape codes.
class Progress {
public:
    Progress(bool enabled, size_t total, const std::string& label) {
        if (!enabled || total == 0 || !stderr_is_tty()) {
            return;
        }
        mBar = std::make_unique<indicators::ProgressBar>(
            indicators::option::BarWidth{40},
            indicators::option::PrefixText{label},
            indicators::option::Start{" ["},
            indicators::option::Fill{"="},
            indicators::option::Lead{">"},
            indicators::option::Remainder{" "},
            indicators::option::End{"]"},
            indicators::option::ShowElapsedTime{true},
            indicators::option::ShowRemainingTime{true},
            indicators::option::ShowPercentage{true},
            indicators::option::MaxProgress{total},
            indicators::option::Stream{std::cerr});
        mTotal = total;
        mEvery = std::max<size_t>(1, total / 500);
    }

    ~Progress() {
        // an exception mid-run would otherwise leave the cursor parked on the bar line
        if (mBar) {
            std::cerr << std::endl;
        }
    }

    void tick() {
        if (!mBar) {
            return;
        }
        ++mStep;
        if (mStep < mTotal && mStep % mEvery == 0) {
            mBar->set_progress(mStep);
        }
    }

    // Reaching MaxProgress makes indicators emit the closing newline itself.
    void finish() {
        if (!mBar) {
            return;
        }
        mBar->set_progress(mTotal);
        mBar.reset();
    }

private:
    std::unique_ptr<indicators::ProgressBar> mBar;
    size_t mTotal = 0;
    size_t mStep = 0;
    size_t mEvery = 1;
};

Metrics run_benchmark(const Options& opt) {
    Metrics m;
    struct stat st;
    if (stat(opt.model_path.c_str(), &st) == 0) {
        m.file_phys_size = static_cast<size_t>(st.st_size);
    }

    // Interpreter Session API. Power_High matches MNN's official benchmark defaults;
    // precision is exposed as --precision (see the header comment).
    MNN::ScheduleConfig config;
    config.type = static_cast<MNNForwardType>(opt.backend);
    config.numThread = opt.threads;
    MNN::BackendConfig backend_config;
    if (opt.precision == "low") {
        backend_config.precision = MNN::BackendConfig::Precision_Low;
    } else if (opt.precision == "normal") {
        backend_config.precision = MNN::BackendConfig::Precision_Normal;
    } else {
        backend_config.precision = MNN::BackendConfig::Precision_High;
    }
    backend_config.power = MNN::BackendConfig::Power_High;
    config.backendConfig = &backend_config;

    std::unique_ptr<MNN::Interpreter, decltype(&MNN::Interpreter::destroy)> net(
        MNN::Interpreter::createFromFile(opt.model_path.c_str()),
        MNN::Interpreter::destroy);
    if (!net) {
        throw std::runtime_error("Failed to load model: " + opt.model_path);
    }
    net->setSessionMode(MNN::Interpreter::Session_Release);
    MNN::Session* session = net->createSession(config);
    if (!session) {
        throw std::runtime_error("Failed to create session: " + opt.model_path);
    }
    MNN::Tensor* input_tensor = net->getSessionInput(session, nullptr);
    MNN::Tensor* output_tensor = net->getSessionOutput(session, nullptr);
    if (!input_tensor || !output_tensor) {
        throw std::runtime_error("Failed to get session input/output");
    }
    // ---- load dataset with MNN::CV ----
    bool do_accuracy = false;
    std::vector<Sample> samples;
    std::vector<std::string> base_dirs;
    if (!opt.image_dir.empty()) {
        samples = load_image_samples(opt);
        base_dirs = {""};
        do_accuracy = !samples.empty();
        m.total = static_cast<int>(samples.size());
    } else if (!opt.dataset_json.empty()) {
        samples = load_samples(opt);
        if (opt.base_dirs.empty()) {
            throw std::runtime_error("--base-dirs is required when using --dataset-json");
        }
        base_dirs = split_csv(opt.base_dirs);
        do_accuracy = !samples.empty();
        m.total = static_cast<int>(samples.size() * base_dirs.size());
    }

    if (!do_accuracy) {
        samples.push_back({"0000_000000000.png", 0});
        base_dirs = {"rgb"};
        m.total = 0;
    }

    auto run_preprocess = [&](const Sample& sample, const std::string& base_dir) {
        std::vector<float> nchw;
        preprocess_to_host(opt, sample, base_dir, nchw);
        void* host = input_tensor->map(MNN::Tensor::MAP_TENSOR_WRITE,
                                       input_tensor->getDimensionType());
        std::copy(nchw.begin(), nchw.end(), static_cast<float*>(host));
        input_tensor->unmap(MNN::Tensor::MAP_TENSOR_WRITE,
                            input_tensor->getDimensionType(), host);
    };

    auto run_inference = [&](const Sample& sample, const std::string& base_dir,
                             std::vector<float>& logits,
                             double& pre_ms, double& infer_ms, double& post_ms) -> int {
        auto t0 = std::chrono::steady_clock::now();
        run_preprocess(sample, base_dir);
        auto t1 = std::chrono::steady_clock::now();
        net->runSession(session);
        auto t2 = std::chrono::steady_clock::now();
        void* host = output_tensor->map(MNN::Tensor::MAP_TENSOR_READ,
                                        output_tensor->getDimensionType());
        int nclasses = output_tensor->elementSize();
        logits.assign(static_cast<float*>(host), static_cast<float*>(host) + nclasses);
        output_tensor->unmap(MNN::Tensor::MAP_TENSOR_READ,
                             output_tensor->getDimensionType(), host);
        auto t3 = std::chrono::steady_clock::now();
        pre_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        infer_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        post_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        return nclasses;
    };

    // ---- memory sampling ----
    std::vector<long> rss_samples;
    auto record_rss = [&]() {
        rss_samples.push_back(get_current_rss_kb());
    };

    // ---- warmup ----
    std::vector<float> logits;
    double pre_ms, infer_ms, post_ms;
    for (int i = 0; i < opt.warmup; ++i) {
        run_inference(samples[0], base_dirs[0], logits, pre_ms, infer_ms, post_ms);
        record_rss();
    }

    // ---- latency + accuracy ----
    // --image-dir mode prints one line per image to stdout, which would fight the bar.
    const bool want_progress = opt.show_progress && opt.image_dir.empty();
    std::vector<double> pre_times, infer_times, post_times, total_times;
    if (do_accuracy) {
        Progress bar(want_progress, samples.size() * base_dirs.size(), "eval");
        for (const auto& bd : base_dirs) {
            for (size_t i = 0; i < samples.size(); ++i) {
                run_inference(samples[i], bd, logits, pre_ms, infer_ms, post_ms);
                pre_times.push_back(pre_ms);
                infer_times.push_back(infer_ms);
                post_times.push_back(post_ms);
                total_times.push_back(pre_ms + infer_ms + post_ms);
                record_rss();
                std::vector<int> top = top_k(logits, 3);
                int y = samples[i].label;
                if (!top.empty() && top[0] == y) ++m.top1;
                if (top.size() >= 3 && std::find(top.begin(), top.end(), y) != top.end()) ++m.top3;
                m.y_true.push_back(y);
                m.y_pred.push_back(top.empty() ? -1 : top[0]);

                if (!opt.image_dir.empty()) {
                    std::vector<int> topk = top_k(logits, opt.topk);
                    printf("[image] %s | true=%d (%s) | pred=",
                        samples[i].image_name.c_str(), y, image_latin_name(samples[i]).c_str());
                    for (size_t t = 0; t < topk.size(); ++t) {
                        printf("%s%d", t ? "," : "", topk[t]);
                    }
                    printf("\n");
                }
                bar.tick();
            }
        }
        bar.finish();
        m.total = static_cast<int>(samples.size() * base_dirs.size());
    } else {
        // latency only on a dummy sample
        Progress bar(want_progress, static_cast<size_t>(opt.repeat), "latency");
        for (int i = 0; i < opt.repeat; ++i) {
            run_inference(samples[0], base_dirs[0], logits, pre_ms, infer_ms, post_ms);
            pre_times.push_back(pre_ms);
            infer_times.push_back(infer_ms);
            post_times.push_back(post_ms);
            total_times.push_back(pre_ms + infer_ms + post_ms);
            record_rss();
            bar.tick();
        }
        bar.finish();
        m.total = 0;
    }

    auto set_stats = [](const std::vector<double>& v, double& mean, double& p50, double& p90, double& p99) {
        if (v.empty()) {
            mean = p50 = p90 = p99 = 0.0;
            return;
        }
        mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        p50 = percentile(v, 0.50);
        p90 = percentile(v, 0.90);
        p99 = percentile(v, 0.99);
    };

    set_stats(pre_times, m.preprocess_ms_mean, m.preprocess_p50, m.preprocess_p90, m.preprocess_p99);
    set_stats(infer_times, m.inference_ms_mean, m.inference_p50, m.inference_p90, m.inference_p99);
    set_stats(post_times, m.postprocess_ms_mean, m.postprocess_p50, m.postprocess_p90, m.postprocess_p99);
    set_stats(total_times, m.latency_ms_mean, m.latency_p50, m.latency_p90, m.latency_p99);

    m.throughput = m.latency_ms_mean > 0.0 ? 1000.0 / m.latency_ms_mean : 0.0;

    m.process_peak_rss_kb = get_peak_rss_kb();
    if (!rss_samples.empty()) {
        m.max_rss_kb = *std::max_element(rss_samples.begin(), rss_samples.end());
        long sum = 0;
        for (long v : rss_samples) sum += v;
        m.mean_rss_kb = sum / static_cast<long>(rss_samples.size());
    } else {
        m.max_rss_kb = m.process_peak_rss_kb;
        m.mean_rss_kb = m.process_peak_rss_kb;
    }
    return m;
}

void write_json(const Options& opt, const std::vector<std::pair<std::string, Metrics>>& results) {
    if (opt.output_json.empty()) return;
    // The label lists are only meaningful when the dataset JSON supplied ground truth,
    // and they are opt-in because they dwarf the rest of the file.
    const bool with_labels = opt.save_pred && !opt.dataset_json.empty();
    nlohmann::json j;
    j["backend"] = opt.backend_name;
    j["precision"] = opt.precision;
    j["threads"] = opt.threads;
    j["warmup"] = opt.warmup;
    j["repeat"] = opt.repeat;
    j["max_images"] = opt.max_images;
    if (with_labels) {
        // y_true/y_pred are concatenated base dir by base dir, so this records how to
        // slice them back into one block per base dir.
        j["base_dirs"] = split_csv(opt.base_dirs);
    }
    j["models"] = nlohmann::json::array();
    for (const auto& [model_path, m] : results) {
        nlohmann::json jm;
        jm["model"] = model_path;
        jm["file_size_bytes"] = m.file_phys_size;
        jm["max_rss_kb"] = m.max_rss_kb;
        jm["mean_rss_kb"] = m.mean_rss_kb;
        jm["process_peak_rss_kb"] = m.process_peak_rss_kb;
        jm["latency_ms_mean"] = m.latency_ms_mean;
        jm["latency_ms_p50"] = m.latency_p50;
        jm["latency_ms_p90"] = m.latency_p90;
        jm["latency_ms_p99"] = m.latency_p99;
        jm["preprocess_ms_mean"] = m.preprocess_ms_mean;
        jm["preprocess_ms_p50"] = m.preprocess_p50;
        jm["preprocess_ms_p90"] = m.preprocess_p90;
        jm["preprocess_ms_p99"] = m.preprocess_p99;
        jm["inference_ms_mean"] = m.inference_ms_mean;
        jm["inference_ms_p50"] = m.inference_p50;
        jm["inference_ms_p90"] = m.inference_p90;
        jm["inference_ms_p99"] = m.inference_p99;
        jm["postprocess_ms_mean"] = m.postprocess_ms_mean;
        jm["postprocess_ms_p50"] = m.postprocess_p50;
        jm["postprocess_ms_p90"] = m.postprocess_p90;
        jm["postprocess_ms_p99"] = m.postprocess_p99;
        jm["throughput_images_per_s"] = m.throughput;
        jm["top1"] = m.top1;
        jm["top3"] = m.top3;
        jm["total"] = m.total;
        if (with_labels && !m.y_true.empty()) {
            jm["y_true"] = m.y_true;
            jm["y_pred"] = m.y_pred;
        }
        j["models"].push_back(jm);
    }
    std::filesystem::path out_path(opt.output_json);
    if (out_path.has_parent_path()) {
        std::filesystem::create_directories(out_path.parent_path());
    }
    std::ofstream f(out_path);
    if (!f) {
        throw std::runtime_error("Cannot open output JSON: " + opt.output_json);
    }
    f << j.dump(2) << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options opt;
        parse_args(argc, argv, opt);

        Metrics m = run_benchmark(opt);
        std::vector<std::pair<std::string, Metrics>> results;
        results.emplace_back(opt.model_path, m);
        printf(
            "model=%s backend=%s precision=%s threads=%d file_size=%.2fMB max_rss=%.1fMB "
            "latency_mean=%.2fms p50=%.2f p90=%.2f p99=%.2f throughput=%.2f "
            "top1=%d/%d top3=%d/%d\n",
            opt.model_path.c_str(), opt.backend_name.c_str(), opt.precision.c_str(), opt.threads,
            m.file_phys_size / 1024.0 / 1024.0, m.max_rss_kb / 1024.0,
            m.latency_ms_mean, m.latency_p50, m.latency_p90, m.latency_p99,
            m.throughput, m.top1, m.total, m.top3, m.total);

        write_json(opt, results);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
#endif
