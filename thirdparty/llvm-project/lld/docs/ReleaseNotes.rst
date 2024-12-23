===========================
lld |release| Release Notes
===========================

.. contents::
    :local:

.. only:: PreRelease

  .. warning::
     These are in-progress notes for the upcoming LLVM |release| release.
     Release notes for previous releases can be found on
     `the Download Page <https://releases.llvm.org/download.html>`_.

Introduction
============

This document contains the release notes for the lld linker, release |release|.
Here we describe the status of lld, including major improvements
from the previous release. All lld releases may be downloaded
from the `LLVM releases web site <https://llvm.org/releases/>`_.

Non-comprehensive list of changes in this release
=================================================

ELF Improvements
----------------

* ``-z nosectionheader`` has been implemented to omit the section header table.
  The operation is similar to ``llvm-objcopy --strip-sections``.
  (`#101286 <https://github.com/llvm/llvm-project/pull/101286>`_)
* Section ``CLASS`` linker script syntax binds input sections to named classes,
  which are referenced later one or more times. This provides access to the
  automatic spilling mechanism of `--enable-non-contiguous-regions` without
  globally changing the semantics of section matching. It also independently
  increases the expressive power of linker scripts.
  (`#95323 <https://github.com/llvm/llvm-project/pull/95323>`_)

Breaking changes
----------------

* Removed support for the (deprecated) `R_RISCV_RVC_LUI` relocation. This
  was a binutils-internal relocation used during relaxation, and was not
  emitted by compilers/assemblers.

COFF Improvements
-----------------
* ``/includeglob`` has been implemented to match the behavior of ``--undefined-glob`` available for ELF.
* ``/lldsavetemps`` allows saving select intermediate LTO compilation results (e.g. resolution, preopt, promote, internalize, import, opt, precodegen, prelink, combinedindex).
* ``/machine:arm64ec`` support completed, enabling the linking of ARM64EC images.
* COFF weak anti-dependency alias symbols are now supported.

MinGW Improvements
------------------
* ``--undefined-glob`` is now supported by translating into the ``/includeglob`` flag.

MachO Improvements
------------------

WebAssembly Improvements
------------------------

Fixes
#####
