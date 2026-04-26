#ifndef FA2_H
#define FA2_H

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cutlass/numeric_types.h>

using namespace cute;

#define kBlockM 64
#define kBlockN 64
#define HeadNum 32
#define HeadDim 64
#define kBlockKSmem (HeadDim % 64 == 0 ? 64 : 32)
#define kSwizzle (kBlockKSmem == 32 ? 2 : 3)
#define ColWarps 4
#define RowWarps 4
#define kNWarps (ColWarps * RowWarps)
#define kNThreads (ColWarps * RowWarps * 32) // 512 threads per block
#define KGmemElemensPerLoad (sizeof(cute::uint128_t) / sizeof(__nv_bfloat16)) // 128/16=8：load 8 elements each time (per thread)
#define kGmemThreadsPerRow (kBlockKSmem / KGmemElemensPerLoad) // 64/8=8: load one row need 8 threads

// global memory 
//
// │◄─T0:  8─►│◄─T1:  8─►│◄─T2:  8─►│   │◄─T7:  8─►│
// ┌──────────┬──────────┬──────────┬───┬──────────┐
// │    T0    │    T1    │    T2    │...│    T7    │
// │    T8    │    T9    │   T10    │...│   T15    │
// │   T16    │   T17    │   T18    │...│   T23    │
// │   T24    │   T25    │   T26    │...│   T31    │
// │    .     │    .     │    .     │   │    .     │
// │    .     │    .     │    .     │   │    .     │
// │   T504   │   T505   │   T506   │...│   T511   │
// └──────────┴──────────┴──────────┴───┴──────────┘
using GmemLayoutAtom = decltype(Layout<Shape<Int<kNThreads/kGmemThreadsPerRow>,Int<kGmemThreadsPerRow>>,// shape(64, 8)
                                Stride<Int<kGmemThreadsPerRow>, _1>>{});
using Gmem_copy_struct = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
using GmemTiledCopyQKV = decltype(make_tiled_copy(Copy_Atom<Gmem_copy_struct, __nv_bfloat16>{}, 
                                                  GmemLayoutAtom{},                             // (m,n) -> thr_idx
                                                  Layout<Shape<_1, _8>>{}));                    // (m,n) -> val_idx
using GmemTiledCopyO = decltype(make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, __nv_bfloat16>{},
                                                GmemLayoutAtom{},
                                                Layout<Shape<_1, _8>>{}));

// shared memory 
// ┌──────────────────────────────┐
// │  SmemLayoutAtom #1           │  ← 8×64
// ├──────────────────────────────┤
// │  SmemLayoutAtom #2           │  ← 8×64
// ├──────────────────────────────┤
// │  SmemLayoutAtom #3           │  ← 8×64
// ├──────────────────────────────┤
// │  SmemLayoutAtom #4           │  ← 8×64
// ├──────────────────────────────┤
// │  SmemLayoutAtom #5           │  ← 8×64
// ├──────────────────────────────┤
// │  SmemLayoutAtom #6           │  ← 8×64
// ├──────────────────────────────┤
// │  SmemLayoutAtom #7           │  ← 8×64
// ├──────────────────────────────┤
// │  SmemLayoutAtom #8           │  ← 8×64
// └──────────────────────────────┘

using SmemLayoutAtomQKV = decltype(composition(Swizzle<kSwizzle, 3, 3>{},          // use composition for swizzle
                                               Layout<Shape<_8, Int<kBlockKSmem>>, // swizzle only support 8 rows, so we have to split 64 rows into 8 groups, each group has 8 rows
                                               Stride<Int<kBlockKSmem>, _1>>{}));
using SmemLayoutAtomO = decltype(composition(Swizzle<kSwizzle, 3, 3>{},
                                             Layout<Shape<_8, Int<kBlockKSmem>>,
                                             Stride<Int<kBlockKSmem>, _1>>{}));

using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQKV{}, 
                                           Shape<Int<kBlockM>, Int<HeadDim>>{}));// 8 groups of 8*64 compose to 64*64
using SmemLayoutKV = decltype(tile_to_shape(SmemLayoutAtomQKV{}, 
                                           Shape<Int<kBlockN>, Int<HeadDim>>{}));
using SmemLayoutVtransposed = decltype(composition(SmemLayoutKV{}, 
                                                   make_layout(Shape<Int<HeadDim>, Int<kBlockN>>{}, 
                                                               GenRowMajor{})));                   // default is col major
using SmemLayoutVtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));
using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{},
                                           Shape<Int<kBlockM>, Int<HeadDim>>{}));


using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, __nv_bfloat16>;
using SmemCopyAtomO = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, __nv_bfloat16>;
using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, __nv_bfloat16>;

// MMA
//                              N=16
//                                |
//                                v
//                          ┌─────┬─────┐
//                 K=16  ─► │ B   │ B   │   
//                  |       └─────┴─────┘
//                  v            
//            ┌─────────┐   ┌───────────┐
//            │ A warp0 │   │ C  Warp0  │  row 0~15
//            ├─────────┤   ├───────────┤
//            │ A warp1 │   │ C  Warp1  │  row 16~31
//  M=64  ─►  ├─────────┤   ├───────────┤
//            │ A warp2 │   │ C  Warp2  │  row 32~47
//            ├─────────┤   ├───────────┤
//            │ A warp3 │   │ C  Warp3  │  row 48~63
//            └─────────┘   └───────────┘

using TiledMma = TiledMMA<MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>, 
                          Layout<Shape<_4,_1,_1>>, // warp arrangement
                          Tile<_64, _16, _16>>;


#endif