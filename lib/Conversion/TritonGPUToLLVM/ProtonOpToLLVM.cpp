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
    auto loc = op.getLoc();
    auto mod = op.getOperation()->getParentOfType<ModuleOp>();
    const int slots = cast<IntegerAttr>(mod->getAttr("proton.slots")).getInt();
    const int numWarpgroup =
        triton::gpu::TritonGPUDialect::getNumWarps(mod) / warpsPerGroup;

    assert(op.getMetric() == triton::ProtonMetric::CYCLE);

    auto bookkeepingLambda = [&](bool isStart,
                                 llvm::SmallVector<Value, 2> &res) {
      Value indexStruct = adaptor.getIndex();
      Value dataStruct = adaptor.getData();

      auto smemPtrTy = ptr_ty(rewriter.getContext(), 3);
      Value smemIndexBasePtr = extract_val(smemPtrTy, indexStruct, 0);
      Value smemDataBasePtr = extract_val(smemPtrTy, dataStruct, 0);

      Value threadId = getThreadId(rewriter, loc);
      Value warpgroupSize =
          i32_val(warpsPerGroup *
                  triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod));
      Value warpgroupId = udiv(threadId, warpgroupSize);
      Value isWarpgroup = icmp_eq(urem(threadId, warpgroupSize), i32_val(0));

      // Load the index from smem
      Value ptr = gep(smemPtrTy, i32_ty, smemIndexBasePtr, warpgroupId);
      Value curIdx =
          targetInfo.loadShared(rewriter, loc, ptr, i32_ty, isWarpgroup);
      Value newIdx = add(curIdx, i32_val(1));
      targetInfo.storeShared(rewriter, loc, ptr, newIdx, isWarpgroup);

      int numWgSlot = slots / numWarpgroup;
      Value wgSlotOffset = mul(warpgroupId, i32_val(wordsPerEntry * numWgSlot));
      Value smemTagOffset =
          add(wgSlotOffset,
              mul(urem(curIdx, i32_val(numWgSlot)), i32_val(wordsPerEntry)));
      Value smemCycleOffset = add(smemTagOffset, i32_val(1));

      // Record the entry
      Value tagPtr = gep(smemPtrTy, i32_ty, smemDataBasePtr, smemTagOffset);
      Value tag = isStart ? i32_val(op.getRegionId())
                          : i32_val(1 << 31 | op.getRegionId());
      targetInfo.storeShared(rewriter, loc, tagPtr, tag, isWarpgroup);

      Value cyclePtr = gep(smemPtrTy, i32_ty, smemDataBasePtr, smemCycleOffset);

      res.push_back(cyclePtr);
      res.push_back(isWarpgroup);
    };

    llvm::SmallVector<Value, 2> res;
    if (op.getIsStart()) {
      // Bookkeeping before getting clock
      bookkeepingLambda(true, res);
      Value cyclePtr = res[0];
      Value isWarpgroup = res[1];
      Value clock = targetInfo.clock(rewriter, loc, false);
      targetInfo.storeShared(rewriter, loc, cyclePtr, clock, isWarpgroup);
    } else {
      // Bookkeeping after getting clock
      Value clock = targetInfo.clock(rewriter, loc, false);
      bookkeepingLambda(false, res);
      Value cyclePtr = res[0];
      Value isWarpgroup = res[1];
      targetInfo.storeShared(rewriter, loc, cyclePtr, clock, isWarpgroup);
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
  const int wordsPerEntry = 2;
  const int warpsPerGroup = 4;
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
    Value dataStruct = adaptor.getData();

    auto loc = op.getLoc();
    auto mod = op.getOperation()->getParentOfType<ModuleOp>();
    const int numWarpgroup =
        triton::gpu::TritonGPUDialect::getNumWarps(mod) / warpsPerGroup;

    // TODO (fywkevin) : Does Triton always have the block size (xxx, 1, 1)?
    Value threadId = getThreadId(rewriter, loc);
    Value isFirstThread = icmp_eq(threadId, i32_val(0));

    // Only warpgroup leader should do these finalize work.
    Block *prevBlock = op->getBlock();
    // Add the 'if' block.
    Block *ifBlock = rewriter.splitBlock(prevBlock, op->getIterator());
    rewriter.setInsertionPointToStart(ifBlock);

    auto gmemPtrTy = ptr_ty(rewriter.getContext(), 1);
    Value gmemBasePtr = adaptor.getPtr();
    auto smemPtrTy = ptr_ty(rewriter.getContext(), 3);

    // Lambda function to load a word from smem and store it to gmem.
    auto copyWord = [&](Value smemStruct, Value smemOffset, Value gmemOffset) {
      Value smemBasePtr = extract_val(smemPtrTy, smemStruct, 0);
      // Load the value from smem
      Value ptr = gep(smemPtrTy, i32_ty, smemBasePtr, smemOffset);
      Value smemLoad =
          targetInfo.loadShared(rewriter, loc, ptr, i32_ty, true_val());
      // Store the value to global memory
      Value gmemPtr = gep(gmemPtrTy, i32_ty, gmemBasePtr, gmemOffset);
      store(smemLoad, gmemPtr);
    };

    int offset = 0;
    const int slots = cast<IntegerAttr>(mod->getAttr("proton.slots")).getInt();
    // scratch: block id (1), sm id (1), index (numWarpgroup), data
    // (proton.slots * wordsPerEntry)
    const int scratchWordSize = 1 + 1 + numWarpgroup + slots * wordsPerEntry;
    Value pidX = targetInfo.programId(rewriter, loc, mod, 0);
    Value pidY = targetInfo.programId(rewriter, loc, mod, 1);
    Value pidZ = targetInfo.programId(rewriter, loc, mod, 2);
    Value smid = targetInfo.smId(rewriter, loc);
    Value gridDimX = rewriter.create<arith::IndexCastOp>(
        loc, i32_ty,
        rewriter.create<::mlir::gpu::GridDimOp>(loc, mlir::gpu::Dimension::x));
    Value gridDimY = rewriter.create<arith::IndexCastOp>(
        loc, i32_ty,
        rewriter.create<::mlir::gpu::GridDimOp>(loc, mlir::gpu::Dimension::y));
    Value pid =
        add(add(pidX, mul(pidY, gridDimX)), mul(pidZ, mul(gridDimX, gridDimY)));
    Value programOffset = mul(i32_val(scratchWordSize), pid);

    // Write back program id. We only support 1D launch.
    Value gmemPidOffset = add(programOffset, i32_val(offset++));
    Value gmemPidPtr = gep(gmemPtrTy, i32_ty, gmemBasePtr, gmemPidOffset);
    store(pid, gmemPidPtr);

    // Write back SM id.
    Value gmemSmOffset = add(programOffset, i32_val(offset++));
    Value gmemSmPtr = gep(gmemPtrTy, i32_ty, gmemBasePtr, gmemSmOffset);
    store(smid, gmemSmPtr);

    // Write back the total counts of entries for each warpgroup.
    for (int i = 0; i < numWarpgroup; i++) {
      Value gmemOffset = add(programOffset, i32_val(offset++));
      Value warpgroupId = i32_val(i);
      copyWord(indexStruct, warpgroupId, gmemOffset);
    }

    // Add the 'else' block and the condition.
    Block *thenBlock = rewriter.splitBlock(ifBlock, op->getIterator());
    rewriter.setInsertionPointToEnd(prevBlock);
    rewriter.create<cf::CondBranchOp>(loc, isFirstThread, ifBlock, thenBlock);

    // Write back the data.
    const int upper = wordsPerEntry * (slots - 1);
    rewriter.setInsertionPointToEnd(ifBlock);
    Value initIdx = rewriter.create<LLVM::ConstantOp>(loc, i32_ty, 0);
    Value wbBaseOffset = add(programOffset, i32_val(offset));

    // TODO (fywkevin): make `loc` precise.
    Block *writeBackBlock = rewriter.createBlock(
        op->getParentRegion(), std::next(Region::iterator(ifBlock)), {i32_ty},
        {loc});
    rewriter.setInsertionPointToStart(writeBackBlock);
    BlockArgument idx = writeBackBlock->getArgument(0);
    Value gmemWbTagOffset = add(wbBaseOffset, idx);
    Value smemTagOffset = idx;
    Value gmemWbCycleOffset = add(gmemWbTagOffset, i32_val(1));
    Value smemCycleOffset = add(smemTagOffset, i32_val(1));
    copyWord(dataStruct, smemTagOffset, gmemWbTagOffset);
    copyWord(dataStruct, smemCycleOffset, gmemWbCycleOffset);
    Value pred = icmp_slt(idx, i32_val(upper));
    Value updatedIdx = add(idx, i32_val(wordsPerEntry));
    rewriter.create<cf::CondBranchOp>(loc, pred, writeBackBlock, updatedIdx,
                                      thenBlock, ArrayRef<Value>());

    rewriter.setInsertionPointToEnd(ifBlock);
    rewriter.create<cf::BranchOp>(loc, writeBackBlock, initIdx);

    rewriter.eraseOp(op);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
  const int wordsPerEntry = 2;
  const int warpsPerGroup = 4;
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
    Value warpgroupSize = i32_val(
        warpsPerGroup * triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod));
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
  const int warpsPerGroup = 4;
};

} // namespace

void mlir::triton::populateProtonOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<LocalRecordOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<ProtonFinalizeOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<ProtonInitOpConversion>(typeConverter, targetInfo, benefit);
}
