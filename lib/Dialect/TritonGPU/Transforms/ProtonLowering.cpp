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
    Operation *m = op->getParentOp();
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
    auto dstType = RankedTensorType::get({1024}, builder.getI32Type());
    Attribute sharedMemorySpace =
        triton::gpu::SharedMemorySpaceAttr::get(context);

    auto CTALayout =
        triton::gpu::CTALayoutAttr::get(context, /*CTAsPerCGA=*/{1},
                                        /*CTASplitNum=*/{1}, /*CTAOrder=*/{0});
    auto encoding =
        triton::gpu::SharedEncodingAttr::get(context, 1, 1, 1, {0}, CTALayout);

    auto smemType =
        MemDescType::get(dstType.getShape(), dstType.getElementType(), encoding,
                         sharedMemorySpace, true);
    Value buffer =
        builder.create<triton::gpu::LocalAllocOp>(m.getLoc(), smemType);
    Value index =
        builder.create<triton::gpu::LocalAllocOp>(m.getLoc(), smemType);

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
