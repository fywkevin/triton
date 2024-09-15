#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace {

using namespace mlir;
using namespace mlir::triton::gpu;

struct LocalRecordOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalRecordOp> {
  explicit LocalRecordOpConversion(LLVMTypeConverter &typeConverter,
                                   const TargetInfoBase &targetInfo,
                                   PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalRecordOp>(typeConverter,
                                                           benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalRecordOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

struct ProtonFinalizeOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::ProtonFinalizeOp> {
  explicit ProtonFinalizeOpConversion(LLVMTypeConverter &typeConverter,
                                      const TargetInfoBase &targetInfo,
                                      PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::ProtonFinalizeOp>(typeConverter,
                                                              benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::ProtonFinalizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value indexStruct = adaptor.getIndex();
    auto loc = op.getLoc();
    auto mod = op.getOperation()->getParentOfType<ModuleOp>();
    const int numWarpgroup =
        triton::gpu::TritonGPUDialect::getNumWarps(mod) / 4;

    // TODO (fywkevin) : check to make sure 1D launch.
    // Get the thread id. We only support 1D launch.
    Value threadId = getThreadId(rewriter, loc);
    Value isFirstThread = icmp_eq(threadId, i32_val(0));

    // Only warpgroup leader should do these finalize work.
    Block *prevBlock = op->getBlock();
    Block *ifBlock = rewriter.splitBlock(prevBlock, op->getIterator());
    rewriter.setInsertionPointToStart(ifBlock);

    auto gmemPtrTy = ptr_ty(rewriter.getContext(), 1);
    Value gmemBasePtr = adaptor.getPtr();
    auto ptrTy = ptr_ty(rewriter.getContext(), 3);
    Value basePtr = extract_val(ptrTy, indexStruct, 0);

    int offset = 0;
    auto slots = cast<IntegerAttr>(mod->getAttr("proton.slots")).getInt();
    // scratch: block id (1), index (numWarpgroup), data (proton.slots * 2)
    const int scratchWordSize = 1 + numWarpgroup + slots * 2;
    // Write back program id. We only support 1D launch.
    Value pid =
        targetInfo.programId(rewriter, loc, op->getParentOfType<ModuleOp>(), 0);
    Value programOffset = mul(i32_val(scratchWordSize), pid);
    Value gmemOffset = add(programOffset, i32_val(offset++));
    Value gmemPtr = gep(gmemPtrTy, i32_ty, gmemBasePtr, gmemOffset);
    store(pid, gmemPtr);

    // Write back the total counts of entries for each warpgroup.
    for (int i = 0; i < numWarpgroup; i++) {
      Value gmemOffset = add(programOffset, i32_val(offset++));
      Value warpgroupId = i32_val(i);
      // Load the index value
      Value ptr = gep(ptrTy, i32_ty, basePtr, warpgroupId);
      Value smemLoad =
          targetInfo.loadShared(rewriter, loc, ptr, i32_ty, true_val());
      // Store the index to global memory
      Value gmemPtr = gep(gmemPtrTy, i32_ty, gmemBasePtr, gmemOffset);
      store(smemLoad, gmemPtr);
    }

    // Write back the data.

    // Split a block after the call.
    Block *thenBlock = rewriter.splitBlock(ifBlock, op->getIterator());
    rewriter.setInsertionPointToEnd(ifBlock);
    rewriter.create<cf::BranchOp>(loc, thenBlock);
    rewriter.setInsertionPointToEnd(prevBlock);
    rewriter.create<cf::CondBranchOp>(loc, isFirstThread, ifBlock, thenBlock);

    rewriter.eraseOp(op);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

struct ProtonInitOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::ProtonInitOp> {
  explicit ProtonInitOpConversion(LLVMTypeConverter &typeConverter,
                                  const TargetInfoBase &targetInfo,
                                  PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::ProtonInitOp>(typeConverter,
                                                          benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::ProtonInitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value indexStruct = adaptor.getIndex();
    auto loc = op.getLoc();
    auto mod = op.getOperation()->getParentOfType<ModuleOp>();

    Value threadId = getThreadId(rewriter, loc);
    Value warpgroupSize =
        i32_val(4 * triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod));
    Value warpgroupId = udiv(threadId, warpgroupSize);
    Value isWarpgroup = icmp_eq(urem(threadId, warpgroupSize), i32_val(0));

    auto ptrTy = ptr_ty(rewriter.getContext(), 3);
    Value basePtr = extract_val(ptrTy, indexStruct, 0);
    Value ptr = gep(ptrTy, i32_ty, basePtr, warpgroupId);
    targetInfo.storeShared(rewriter, loc, ptr, i32_val(0), isWarpgroup);
    rewriter.eraseOp(op);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::populateProtonOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<LocalRecordOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<ProtonFinalizeOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<ProtonInitOpConversion>(typeConverter, targetInfo, benefit);
}
