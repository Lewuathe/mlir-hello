; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin21.4.0"

@nl = internal constant [2 x i8] c"\0A\00"
@frmt_spec = internal constant [4 x i8] c"%f \00"

declare i8* @malloc(i64)

declare void @free(i8*)

declare i32 @printf(i8*, ...)

define void @main() !dbg !3 {
  %1 = call i8* @malloc(i64 ptrtoint (double* getelementptr (double, double* null, i64 6) to i64)), !dbg !7
  %2 = bitcast i8* %1 to double*, !dbg !7
  %3 = insertvalue { double*, double*, i64, [2 x i64], [2 x i64] } undef, double* %2, 0, !dbg !7
  %4 = insertvalue { double*, double*, i64, [2 x i64], [2 x i64] } %3, double* %2, 1, !dbg !7
  %5 = insertvalue { double*, double*, i64, [2 x i64], [2 x i64] } %4, i64 0, 2, !dbg !7
  %6 = insertvalue { double*, double*, i64, [2 x i64], [2 x i64] } %5, i64 2, 3, 0, !dbg !7
  %7 = insertvalue { double*, double*, i64, [2 x i64], [2 x i64] } %6, i64 3, 3, 1, !dbg !7
  %8 = insertvalue { double*, double*, i64, [2 x i64], [2 x i64] } %7, i64 3, 4, 0, !dbg !7
  %9 = insertvalue { double*, double*, i64, [2 x i64], [2 x i64] } %8, i64 1, 4, 1, !dbg !7
  %10 = extractvalue { double*, double*, i64, [2 x i64], [2 x i64] } %9, 1, !dbg !7
  %11 = getelementptr double, double* %10, i64 0, !dbg !7
  store double 1.000000e+00, double* %11, align 8, !dbg !7
  %12 = extractvalue { double*, double*, i64, [2 x i64], [2 x i64] } %9, 1, !dbg !7
  %13 = getelementptr double, double* %12, i64 1, !dbg !7
  store double 2.000000e+00, double* %13, align 8, !dbg !7
  %14 = extractvalue { double*, double*, i64, [2 x i64], [2 x i64] } %9, 1, !dbg !7
  %15 = getelementptr double, double* %14, i64 2, !dbg !7
  store double 3.000000e+00, double* %15, align 8, !dbg !7
  %16 = extractvalue { double*, double*, i64, [2 x i64], [2 x i64] } %9, 1, !dbg !7
  %17 = getelementptr double, double* %16, i64 3, !dbg !7
  store double 4.000000e+00, double* %17, align 8, !dbg !7
  %18 = extractvalue { double*, double*, i64, [2 x i64], [2 x i64] } %9, 1, !dbg !7
  %19 = getelementptr double, double* %18, i64 4, !dbg !7
  store double 5.000000e+00, double* %19, align 8, !dbg !7
  %20 = extractvalue { double*, double*, i64, [2 x i64], [2 x i64] } %9, 1, !dbg !7
  %21 = getelementptr double, double* %20, i64 5, !dbg !7
  store double 6.000000e+00, double* %21, align 8, !dbg !7
  br label %22, !dbg !9

22:                                               ; preds = %37, %0
  %23 = phi i64 [ 0, %0 ], [ %39, %37 ]
  %24 = icmp slt i64 %23, 2, !dbg !9
  br i1 %24, label %25, label %40, !dbg !9

25:                                               ; preds = %22
  br label %26, !dbg !9

26:                                               ; preds = %29, %25
  %27 = phi i64 [ 0, %25 ], [ %36, %29 ]
  %28 = icmp slt i64 %27, 3, !dbg !9
  br i1 %28, label %29, label %37, !dbg !9

29:                                               ; preds = %26
  %30 = extractvalue { double*, double*, i64, [2 x i64], [2 x i64] } %9, 1, !dbg !9
  %31 = mul i64 %23, 3, !dbg !9
  %32 = add i64 %31, %27, !dbg !9
  %33 = getelementptr double, double* %30, i64 %32, !dbg !9
  %34 = load double, double* %33, align 8, !dbg !9
  %35 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), double %34), !dbg !9
  %36 = add i64 %27, 1, !dbg !9
  br label %26, !dbg !9

37:                                               ; preds = %26
  %38 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @nl, i64 0, i64 0)), !dbg !9
  %39 = add i64 %23, 1, !dbg !9
  br label %22, !dbg !9

40:                                               ; preds = %22
  %41 = extractvalue { double*, double*, i64, [2 x i64], [2 x i64] } %9, 0, !dbg !7
  %42 = bitcast double* %41 to i8*, !dbg !7
  call void @free(i8* %42), !dbg !7
  ret void, !dbg !10
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "mlir", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "LLVMDialectModule", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "main", linkageName: "main", scope: null, file: !4, line: 5, type: !5, scopeLine: 5, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!4 = !DIFile(filename: "test/Hello/print.mlir", directory: "/Users/sasakikai/dev/mlir-hello")
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = !DILocation(line: 6, column: 10, scope: !8)
!8 = !DILexicalBlockFile(scope: !3, file: !4, discriminator: 0)
!9 = !DILocation(line: 7, column: 5, scope: !8)
!10 = !DILocation(line: 8, column: 5, scope: !8)

