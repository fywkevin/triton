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
    auto ptrTy = ptr_ty(rewriter.getContext(), 3);
    Value ptr = extract_val(ptrTy, indexStruct, 0);
    targetInfo.storeShared(rewriter, loc, ptr, i32_val(0), true_val());
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
