// RUN: hello-opt %s | FileCheck %s

// CHECK: @hello_word_string = internal constant [16 x i8] c"Hello, World! \0A\00"
// CHECK: define void @main()
func.func @main() {
    // CHECK: %{{.*}} = call i32 (ptr, ...) @printf(ptr @hello_word_string)
    "hello.world"() : () -> ()
    return
}
