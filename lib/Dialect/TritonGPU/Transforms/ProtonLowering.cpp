#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

// TODO (fywkevin) : clean up the headers
namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUPROTONLOWERING
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class ProtonRecordOpLowering : public OpRewritePattern<ProtonRecordOp> {
public:
  ProtonRecordOpLowering(MLIRContext *ctx, Value buf, Value idx)
      : OpRewritePattern::OpRewritePattern(ctx), buffer(buf), index(idx) {}

  LogicalResult matchAndRewrite(ProtonRecordOp op,
                                PatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();
    rewriter.replaceOpWithNewOp<LocalRecordOp>(
        op, buffer, index, op.getIsStart(), op.getRegionId(), op.getMetric());
    return success();
  }

private:
  Value buffer = nullptr;
  Value index = nullptr;
};

class TritonGPUProtonLoweringPass
    : public impl::TritonGPUProtonLoweringBase<TritonGPUProtonLoweringPass> {
public:
  TritonGPUProtonLoweringPass() = default;

  void runOnOperation() override {
    ModuleOp m = getOperation();
    MLIRContext *context = m.getContext();

    // TODO (fywkevin) : better way to report failure.
    FuncOp func;
    for (auto op : m.getOps<triton::FuncOp>()) {
      if (!func)
        func = op;
      else
        llvm::report_fatal_error("only expect one function in the module");
    }

    bool noinline = cast<BoolAttr>(func->getAttr("noinline")).getValue();
    assert(!noinline &&
           "only support inlined triton kernels for now (noinline = false)");

    assert(m->hasAttr("proton.slots") && "intra-kernel profiling not enabled");
    auto slots = cast<IntegerAttr>(m->getAttr("proton.slots")).getInt();

    Location loc = func.getLoc();

    //===--------------------------------------------------------------------===//
    // Allocate shared memory resources.
    //===--------------------------------------------------------------------===//

    OpBuilder builder(context);
    builder.setInsertionPointToStart(&func.getBody().front());

    // Alloc the shared memory for buffer (uninitialized).
    Attribute sharedMemorySpace =
        triton::gpu::SharedMemorySpaceAttr::get(context);
    auto ctaLayout =
        triton::gpu::CTALayoutAttr::get(context, /*CTAsPerCGA=*/{1},
                                        /*CTASplitNum=*/{1}, /*CTAOrder=*/{0});
    auto encoding =
        triton::gpu::SharedEncodingAttr::get(context, 1, 1, 1, {0}, ctaLayout);
    auto bufferType =
        MemDescType::get({slots}, builder.getI32Type(), encoding,
                         sharedMemorySpace, /*mutable_memory=*/true);
    Value buffer = builder.create<triton::gpu::LocalAllocOp>(loc, bufferType);

    // Alloc the shared memory for index (uninitialized).
    int numWarpgroups = TritonGPUDialect::getNumWarps(m) / 4;
    auto indexType =
        MemDescType::get({numWarpgroups}, builder.getI32Type(), encoding,
                         sharedMemorySpace, /*mutable_memory=*/true);
    Value index = builder.create<triton::gpu::LocalAllocOp>(loc, indexType);

    //===--------------------------------------------------------------------===//
    // Insert and lower Proton operators.
    //===--------------------------------------------------------------------===//

    builder.setInsertionPointAfter(index.getDefiningOp());
    builder.create<ProtonInitOp>(loc, index);

    mlir::RewritePatternSet patterns(context);
    patterns.add<ProtonRecordOpLowering>(context, buffer, index);
    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed())
      signalPassFailure();

    Operation *ret = &func.getBody().front().back();
    builder.setInsertionPoint(ret);
    Value profileMem = func.getArguments().back();
    builder.create<ProtonFinalizeOp>(loc, buffer, index, profileMem);
    return;
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
