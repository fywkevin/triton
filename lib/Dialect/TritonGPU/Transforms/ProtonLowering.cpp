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
    // TODO (fywkevin) : clean-up the loc.
    FuncOp func;
    for (auto op : m.getOps<triton::FuncOp>()) {
      if (!func)
        func = op;
      else
        llvm::report_fatal_error("only expect one function in the module");
    }

    // Allocate shared memory resource.
    OpBuilder builder(context);
    builder.setInsertionPointToStart(&func.getBody().front());
    auto bufType = RankedTensorType::get({1024}, builder.getI32Type());
    Attribute sharedMemorySpace =
        triton::gpu::SharedMemorySpaceAttr::get(context);

    auto ctaLayout =
        triton::gpu::CTALayoutAttr::get(context, /*CTAsPerCGA=*/{1},
                                        /*CTASplitNum=*/{1}, /*CTAOrder=*/{0});
    auto encoding =
        triton::gpu::SharedEncodingAttr::get(context, 1, 1, 1, {0}, ctaLayout);

    auto smemType =
        MemDescType::get(bufType.getShape(), bufType.getElementType(), encoding,
                         sharedMemorySpace, /*mutable_memory=*/true);
    Value buffer =
        builder.create<triton::gpu::LocalAllocOp>(m.getLoc(), smemType);

    auto indexTensorType = RankedTensorType::get({1}, builder.getI32Type());
    auto idxSmemType = MemDescType::get(
        indexTensorType.getShape(), indexTensorType.getElementType(), encoding,
        sharedMemorySpace, /*mutable_memory=*/true);
    auto scalar = builder.create<arith::ConstantOp>(
        m.getLoc(), builder.getI32Type(), builder.getI32IntegerAttr(0));
    auto splat =
        builder.create<triton::SplatOp>(m.getLoc(), indexTensorType, scalar);
    Value index = builder.create<triton::gpu::LocalAllocOp>(m.getLoc(),
                                                            idxSmemType, splat);

    // Rewrite the record op with resource binded.
    mlir::RewritePatternSet patterns(context);
    patterns.add<ProtonRecordOpLowering>(context, buffer, index);
    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed())
      signalPassFailure();

    // Write back to the global memory.
    return;
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
