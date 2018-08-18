#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <topi/broadcast.h>
#include <topi/elemwise.h>
#include <topi/nn.h>
#include <topi/nn/pooling.h>
#include <topi/transform.h>
#include <topi/x86/default.h>
#include <tvm/build_module.h>
#include <tvm/tvm.h>
#include "ndarray.hpp"
#include "shape.hpp"
template <typename T>
bool all_close(const std::vector<T>& a,
               const std::vector<T>& b,
               T rtol = static_cast<T>(1e-5),
               T atol = static_cast<T>(1e-8))
{
    bool rc = true;
    assert(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i)
    {
        if (std::abs(a[i] - b[i]) > atol + rtol * std::abs(b[i]))
        {
            std::cout << a[i] << " is not close to " << b[i] << " at index " << i << std::endl;
            rc = false;
        }
    }
    return rc;
}
int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    testing::FLAGS_gtest_death_test_style = "threadsafe";
    return RUN_ALL_TESTS();
}
static const DLDataType DLType_Float32{kDLFloat, 32, 1};
DLTensor create_dltensor(const DLDataType& type, const size_t ndim, tvm_index_t* shape, void* data)
{
    DLContext m_dl_ctx;
    m_dl_ctx.device_type = static_cast<DLDeviceType>(kDLCPU);
    m_dl_ctx.device_id = 0;
    DLTensor t;
    t.ctx = m_dl_ctx;
    t.ndim = ndim;
    t.dtype = type;
    t.shape = static_cast<int64_t*>(shape);
    t.strides = nullptr;
    t.byte_offset = 0;
    t.data = data;
    return t;
}
void call_func(tvm::runtime::PackedFunc func,
               std::vector<float>& at,
               std::vector<float>& bt,
               std::vector<float>& rt,
               std::vector<int64_t>& ashape,
               std::vector<int64_t>& bshape,
               std::vector<int64_t>& rshape)
{
    DLTensor a = create_dltensor(DLType_Float32, ashape.size(), &ashape[0], at.data());
    DLTensor b = create_dltensor(DLType_Float32, bshape.size(), &bshape[0], bt.data());
    DLTensor r = create_dltensor(DLType_Float32, rshape.size(), &rshape[0], rt.data());
    func(&a, &b, &r);
}
void call_func(tvm::runtime::PackedFunc func,
               std::vector<float>& at,
               std::vector<float>& rt,
               std::vector<int64_t>& ashape,
               std::vector<int64_t>& rshape)
{
    DLTensor a = create_dltensor(DLType_Float32, ashape.size(), &ashape[0], at.data());
    DLTensor r = create_dltensor(DLType_Float32, rshape.size(), &rshape[0], rt.data());
    func(&a, &r);
}
tvm::Module g_mod;
const tvm::runtime::PackedFunc get_func(const tvm::Array<tvm::Tensor>& G)
{
    auto m_config = tvm::build_config();
    auto m_target = tvm::target::llvm();
    std::unordered_map<tvm::Tensor, tvm::Buffer> binds;
    auto schedule = topi::x86::default_schedule(m_target, {G[G.size() - 1]});
    auto lowered = tvm::lower(schedule, G, "func", binds, m_config);
    g_mod = tvm::build(lowered, m_target, tvm::Target(), m_config);
    return g_mod->GetFunction("func", false);
}

void copy_data(std::vector<float>& a, const std::vector<float>& v)
{
    a = v;
}
template <typename T>
std::vector<float>& read_vector(std::vector<T>& v)
{
    return v;
}

#define BINARY_FUNC(ashape, bshape)                                                                \
    tvm::Array<tvm::Expr> ae;                                                                      \
    tvm::Array<tvm::Expr> be;                                                                      \
    for (auto& v : ashape)                                                                         \
        ae.push_back(tvm::make_const(tvm::Int(32), v));                                            \
    for (auto& v : bshape)                                                                         \
        be.push_back(tvm::make_const(tvm::Int(32), v));                                            \
    auto A = tvm::placeholder(ae, tvm::Float(32), "a");                                            \
    auto B = tvm::placeholder(be, tvm::Float(32), "b");

TEST(Tensor, div)
{
    using namespace std;

    vector<int64_t> shape{2, 2};
    BINARY_FUNC(shape, shape)
    auto R = topi::divide(A, B, "tensor", topi::kBroadcast);

    vector<float> a, b;
    copy_data(a, vector<float>{2, 4, 8, 16});
    copy_data(b, vector<float>{1, 2, 4, 8});

    vector<float> result(4);
    call_func(get_func({A, B, R}), a, b, result, shape, shape, shape);

    EXPECT_EQ((vector<float>{2, 2, 2, 2}), read_vector<float>(result));
}

TEST(Tensor, dot)
{
    using namespace std;

    vector<int64_t> shape{2, 2};
    BINARY_FUNC(shape, shape)
    auto R = topi::matmul(A, B, false, false, "tensor", topi::kMatMul);

    vector<float> a, b;
    copy_data(a, vector<float>{1, 2, 3, 4});
    copy_data(b, vector<float>{5, 6, 7, 8});

    vector<float> result(4);
    call_func(get_func({A, B, R}), a, b, result, shape, shape, shape);

    EXPECT_EQ((vector<float>{19, 22, 43, 50}), read_vector<float>(result));
}
TEST(Tensor, maxpool)
{
    using namespace std;
    using namespace ngraph;
    vector<int64_t> shape{2, 2, 5, 5};
    vector<int64_t> rshape{2, 2, 4, 3};
    vector<size_t> k{2, 3}, s{1, 1}, p{0, 0, 0, 0};
    tvm::Array<tvm::Expr> ae;
    for (auto& v : shape)
        ae.push_back(tvm::make_const(tvm::Int(32), v));
    auto I = tvm::placeholder(ae, tvm::Float(32), "I");
    auto R = topi::nn::pool(I, {2, 3}, {1, 1}, {0, 0, 0, 0}, topi::nn::kMaxPool, false);
    auto func = get_func({I, R});

    auto at = test::NDArray<float, 4>({{{{0, 1, 0, 2, 1}, // img 0 chan 0
                                         {0, 3, 2, 0, 0},
                                         {2, 0, 0, 0, 1},
                                         {2, 0, 1, 1, 2},
                                         {0, 2, 1, 0, 0}},

                                        {{0, 0, 0, 2, 0}, // img 0 chan 1
                                         {0, 2, 3, 0, 1},
                                         {2, 0, 1, 0, 2},
                                         {3, 1, 0, 0, 0},
                                         {2, 0, 0, 0, 0}}},

                                       {{{0, 2, 1, 1, 0}, // img 1 chan 0
                                         {0, 0, 2, 0, 1},
                                         {0, 0, 1, 2, 3},
                                         {2, 0, 0, 3, 0},
                                         {0, 0, 0, 0, 0}},

                                        {{2, 1, 0, 0, 1}, // img 1 chan 1
                                         {0, 2, 0, 0, 0},
                                         {1, 1, 2, 0, 2},
                                         {1, 1, 1, 0, 1},
                                         {1, 0, 0, 0, 2}}}})
                  .get_vector();
    auto rtv = test::NDArray<float, 4>({{{{3, 3, 2}, // img 0 chan 0
                                          {3, 3, 2},
                                          {2, 1, 2},
                                          {2, 2, 2}},

                                         {{3, 3, 3}, // img 0 chan 1
                                          {3, 3, 3},
                                          {3, 1, 2},
                                          {3, 1, 0}}},

                                        {{{2, 2, 2}, // img 1 chan 0
                                          {2, 2, 3},
                                          {2, 3, 3},
                                          {2, 3, 3}},

                                         {{2, 2, 1}, // img 1 chan 1
                                          {2, 2, 2},
                                          {2, 2, 2},
                                          {1, 1, 2}}}})
                   .get_vector();
    vector<float> rt(rtv.size());

    call_func(func, at, rt, shape, rshape);
    EXPECT_EQ(rtv, rt);
}
TEST(Tensor, conv)
{
    using namespace std;
    using namespace ngraph;
    vector<int64_t> shape{2, 1, 3, 5};
    vector<int64_t> bshape{2, 1, 2, 2};
    vector<int64_t> rshape{2, 1, 2, 4};
    tvm::Array<tvm::Expr> ae, be;
    for (auto& v : shape)
        ae.push_back(tvm::make_const(tvm::Int(32), v));
    for (auto& v : bshape)
        be.push_back(tvm::make_const(tvm::Int(32), v));
    auto I = tvm::placeholder(ae, tvm::Float(32), "I");
    auto W = tvm::placeholder(be, tvm::Float(32), "W");
    auto R = topi::conv2d_nchw(I, W);
    std::cout << R->shape << std::endl;
    auto func = get_func({I, W, R});

    // Create some tensors for input/output
    vector<float> at, bt;
    copy_data(at,
              vector<float>{0.67187500f,  0.54687500f,  -0.56250000f, -0.35937500f, -0.09375000f,
                            0.54687500f,  -0.54687500f, 0.89062500f,  0.82812500f,  -0.54687500f,
                            1.00000000f,  -0.07812500f, -0.89062500f, 0.40625000f,  -0.35937500f,
                            0.54687500f,  0.60937500f,  0.59375000f,  0.09375000f,  -0.21875000f,
                            0.76562500f,  0.40625000f,  -0.73437500f, -0.95312500f, -0.50000000f,
                            -0.29687500f, 0.76562500f,  -0.26562500f, -0.50000000f, 0.53125000f});
    copy_data(bt,
              vector<float>{0.67187500f,
                            0.54687500f,
                            -0.56250000f,
                            -0.35937500f,
                            -0.09375000f,
                            0.54687500f,
                            -0.54687500f,
                            0.89062500f});

    vector<float> rtv{0.63940430f,  0.04736328f,  -1.37304688f, -0.56201172f, -0.46606445f,
                      0.48364258f,  1.40625000f,  0.15795898f,  -0.55004883f, 0.73339844f,
                      0.10668945f,  -0.95751953f, -0.96679688f, -0.21215820f, 1.21826172f,
                      -0.91894531f, 0.12402344f,  0.76953125f,  1.20581055f,  0.65917969f,
                      0.62841797f,  -0.46386719f, -0.68554688f, -0.82348633f, 0.22509766f,
                      -0.60864258f, -0.45166016f, -0.05249023f, 0.99462891f,  -1.09497070f,
                      -0.75244141f, 0.56250000f};

    vector<float> rt(rtv.size());
    call_func(func, at, bt, rt, shape, bshape, rshape);
    EXPECT_TRUE(
        all_close<float>(vector<float>{rtv}, read_vector<float>(rt), 1.0e-4f, 1.0e-6f));
}
