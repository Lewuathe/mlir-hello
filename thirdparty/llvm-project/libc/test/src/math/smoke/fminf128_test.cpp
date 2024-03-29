//===-- Unittests for fminf128 --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FMinTest.h"

#include "src/math/fminf128.h"

LIST_FMIN_TESTS(float128, LIBC_NAMESPACE::fminf128)
