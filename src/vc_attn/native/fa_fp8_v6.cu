// Independent SM103 attention experiment. CUDA Toolkit + inline PTX only.
// Hardware encodings follow NVIDIA's PTX ISA, tcgen05 matrix descriptors.
// H3 numerical policy follows flash_attention_plus (descale, FP8 P, deadband).
// Independent FP8 scheduling and register allocation; BF16 output conversion only.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstddef>
#include <string>
#include <cmath>
#include "tmem_ops.cuh"

#ifndef RAW_VARIANT
#define RAW_VARIANT fa_fp8_v6
#endif
#ifndef SOFTMAX_REGS
#define SOFTMAX_REGS 160
#endif
#ifndef OTHER_REGS
#define OTHER_REGS 80
#endif
#ifndef CORRECTION_REGS
#define CORRECTION_REGS 112
#endif
static_assert(SOFTMAX_REGS % 8 == 0 && CORRECTION_REGS % 8 == 0);
static_assert(8 * SOFTMAX_REGS + 4 * CORRECTION_REGS + 4 * OTHER_REGS <= 2048);
namespace RAW_VARIANT {

__device__ __forceinline__ unsigned shared_addr(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mb_init(unsigned p, unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(p), "r"(count) : "memory");
}
__device__ __forceinline__ void arrive(unsigned p) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"(p) : "memory");
}
__device__ __forceinline__ void wait(unsigned p, unsigned phase) {
  asm volatile("{ .reg .pred done; WAIT: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 done, [%0], %1, 10000000; @!done bra WAIT; }"
               :: "r"(p), "r"(phase) : "memory");
}
__device__ __forceinline__ void expect(unsigned p, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
               :: "r"(p), "r"(bytes) : "memory");
}
__device__ __forceinline__ void commit(unsigned p) {

    asm volatile("{ .reg .pred elected; elect.sync _|elected, 0xffffffff; @elected tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0]; }" :: "r"(p) : "memory");

}
__device__ __forceinline__ void fence_st() {
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void fence_ld() {
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void fence_after() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void fence_before() {
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void tma(const CUtensorMap* map, unsigned dst,
                                  unsigned barrier, int d, int h, int s) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
               :: "r"(dst), "l"(map), "r"(d), "r"(h), "r"(s), "r"(barrier) : "memory");
}
__device__ __forceinline__ void tma_expect(const CUtensorMap* map, unsigned dst,
                                  unsigned barrier, int d, int h, int s,unsigned bytes) {
  asm volatile("{ .reg .pred elected; elect.sync _|elected, 0xffffffff;"
    " @elected mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%5], %6;"
    " @elected cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5]; }"
    :: "r"(dst), "l"(map), "r"(d), "r"(h), "r"(s), "r"(barrier),"r"(bytes) : "memory");
}
// 128-byte swizzle: eight contiguous 16-byte groups, eight rows per repeat.
__device__ __forceinline__ uint64_t descriptor(unsigned ptr, unsigned leading = 16) {
  return (uint64_t(ptr >> 4) & 0x3fff) | (uint64_t(leading >> 4) << 16)
       | (uint64_t(1024 >> 4) << 32) | (uint64_t(1) << 46) | (uint64_t(2) << 61);
}

__device__ __forceinline__ void mma_qk(unsigned dst, unsigned q, unsigned k) {
  constexpr unsigned id=(1u<<4)|(16u<<17)|(8u<<24)|0u;
  uint64_t ad=descriptor(q),bd=descriptor(k);


    asm volatile("{ .reg .pred elected; elect.sync _|elected, 0xffffffff; .reg .b64 a,b; .reg .b32 z; .reg .pred p; mov.b64 a,%1; mov.b64 b,%2; mov.b32 z,0; setp.ne.u32 p,z,z;\n"
      "@elected tcgen05.mma.cta_group::1.kind::f8f6f4 [%0],a,b,%3,{z,z,z,z},p;\n"
      "add.u64 a,a,2; add.u64 b,b,2; setp.eq.u32 p,z,z;\n"
      "@elected tcgen05.mma.cta_group::1.kind::f8f6f4 [%0],a,b,%3,{z,z,z,z},p;\n"
      "add.u64 a,a,2; add.u64 b,b,2; setp.eq.u32 p,z,z;\n"
      "@elected tcgen05.mma.cta_group::1.kind::f8f6f4 [%0],a,b,%3,{z,z,z,z},p;\n"
      "add.u64 a,a,2; add.u64 b,b,2; setp.eq.u32 p,z,z;\n"
      "@elected tcgen05.mma.cta_group::1.kind::f8f6f4 [%0],a,b,%3,{z,z,z,z},p;\n"
      "}" :: "r"(dst),"l"(ad),"l"(bd),"r"(id) : "memory");

}
template<int Half> __device__ __forceinline__ void mma_pv(unsigned dst,unsigned p,unsigned v,bool accumulate) {
  constexpr unsigned id=(1u<<4)|(16u<<17)|(8u<<24)|(1u<<16)|0u;
  uint64_t bd=descriptor(v+Half*64*128,16);
  unsigned pa=p+Half*16,acc=accumulate || Half!=0;


    asm volatile("{ .reg .pred elected; elect.sync _|elected, 0xffffffff; .reg .b64 b; .reg .b32 a,z; .reg .pred p; mov.b32 a,%1; mov.b64 b,%2; mov.b32 z,0; setp.ne.u32 p,%4,0;\n"
      "@elected tcgen05.mma.cta_group::1.kind::f8f6f4 [%0],[a],b,%3,{z,z,z,z},p;\n"
      "add.u32 a,a,8; add.u64 b,b,256; setp.eq.u32 p,z,z;\n"
      "@elected tcgen05.mma.cta_group::1.kind::f8f6f4 [%0],[a],b,%3,{z,z,z,z},p;\n"
      "}" :: "r"(dst),"r"(pa),"l"(bd),"r"(id),"r"(acc) : "memory");

}
__device__ __forceinline__ float exp2_fast(float x) {
  float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y;
}
__device__ __forceinline__ void fma2(float& x, float& y, float a, float b) {
  asm("{ .reg .b64 xy, aa, bb, oo; mov.b64 xy, {%2,%3}; mov.b64 aa, {%4,%4}; mov.b64 bb, {%5,%5}; fma.rn.f32x2 oo, xy, aa, bb; mov.b64 {%0,%1}, oo; }"
      : "=f"(x), "=f"(y) : "f"(x), "f"(y), "f"(a), "f"(b));
}
__device__ __forceinline__ void add2(float& a,float& b,float c,float d) {
  asm("{ .reg .b64 x,y,z; mov.b64 x,{%2,%3}; mov.b64 y,{%4,%5}; add.f32x2 z,x,y; mov.b64 {%0,%1},z; }"
      : "=f"(a),"=f"(b) : "f"(a),"f"(b),"f"(c),"f"(d));
}
__device__ __forceinline__ float pack4(float a, float b, float c, float d) {
  float packed;
  asm("{ .reg .b16 lo, hi; cvt.rn.satfinite.e4m3x2.f32 lo, %2, %1; cvt.rn.satfinite.e4m3x2.f32 hi, %4, %3; mov.b32 %0, {lo,hi}; }"
      : "=f"(packed) : "f"(a), "f"(b), "f"(c), "f"(d));
  return packed;
}
__device__ __forceinline__ float pack2(float a, float b) {
  float packed;
  asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=f"(packed) : "f"(a), "f"(b));
  return packed;
}

struct Barriers {
  uint64_t q, kv_full[4], kv_empty[4];
  uint64_t scores[2], stats_full[2], stats_empty[2], p_ready[2], p_tail[2], o_done[2], final[2];
  unsigned tmem;
};
// Explicit offsets below keep hot barrier addressing out of polling loops.
static_assert(offsetof(Barriers,q)==0 && offsetof(Barriers,kv_full)==8 && offsetof(Barriers,kv_empty)==40);
static_assert(offsetof(Barriers,scores)==72 && offsetof(Barriers,stats_full)==88 && offsetof(Barriers,stats_empty)==104);
static_assert(offsetof(Barriers,p_ready)==120 && offsetof(Barriers,p_tail)==136 && offsetof(Barriers,o_done)==152);
static_assert(offsetof(Barriers,final)==168 && offsetof(Barriers,tmem)==184);
struct Params {
  CUtensorMap q, k, v;
  __nv_bfloat16* out;
  float* lse;
  float* debug;
  const float *qd, *kd, *vd;
  int s, h;
};
template<bool Scaled> __global__ __launch_bounds__(512, 1) void attention_fp8(const __grid_constant__ Params p) {
  constexpr int bytes = 16384 * 1;
  extern __shared__ __align__(1024) unsigned char sm[];
  auto& b = *reinterpret_cast<Barriers*>(sm + 6 * bytes + 32768);
  unsigned qbase = shared_addr(sm), kvbase = qbase + 2 * bytes;
  unsigned row = threadIdx.x & 127, warp = threadIdx.x / 32, lane = threadIdx.x & 31;
  // Publish warp-role uniformity once; avoids divergent loop bookkeeping.
  warp=__shfl_sync(0xffffffff,warp,0);
  int head = blockIdx.y, qblock = blockIdx.x * 2, blocks = (p.s + 127) / 128;
  if (threadIdx.x == 0) {
    mb_init(shared_addr(&b.q), 1);
#pragma unroll
    for (int i=0;i<4;++i) { mb_init(shared_addr(&b.kv_full[i]),1); mb_init(shared_addr(&b.kv_empty[i]),1); }
#pragma unroll
    for (int i=0;i<2;++i) {
      mb_init(shared_addr(&b.scores[i]),1); mb_init(shared_addr(&b.stats_full[i]),128);
      mb_init(shared_addr(&b.stats_empty[i]),128); mb_init(shared_addr(&b.p_ready[i]),256);
      mb_init(shared_addr(&b.p_tail[i]),128); mb_init(shared_addr(&b.o_done[i]),1);
      mb_init(shared_addr(&b.final[i]),128);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == 12) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"(shared_addr(&b.tmem)) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
  }
  __syncthreads();
  unsigned tm = b.tmem;
  unsigned barrier_base=shared_addr(&b);
  asm volatile("mov.b32 %0,%0;" : "+r"(barrier_base));
  if (warp >= 12) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(OTHER_REGS) : "memory");
    if (warp == 13) {
      if(lane==0) {
      expect((barrier_base+0), 2 * bytes);
#pragma unroll
      for (int stage=0;stage<2;++stage) {
        tma(&p.q, qbase + stage*bytes, (barrier_base+0), 0, head, (qblock+stage)*128);

      }
      }
      for (int n=0;n<blocks;++n) {
#pragma unroll
        for(int which=0;which<2;++which) {
          int slot = (n & 1)*2 + which;
          if(n>=2) wait((barrier_base+40+8*(slot)), ((n/2)-1)&1);
          tma_expect(which ? &p.v : &p.k,kvbase+slot*bytes,(barrier_base+8+8*(slot)),0,head,n*128,bytes);

        }
      }
    }
    // Full-warp control keeps MMA descriptors uniform; PTX elects the issuer.
    if (warp == 12) {
      qbase=__shfl_sync(0xffffffff,qbase,0);
      kvbase=__shfl_sync(0xffffffff,kvbase,0);
      tm=__shfl_sync(0xffffffff,tm,0);
      wait((barrier_base+0),0); wait((barrier_base+8+8*(0)),0); fence_after();
      mma_qk(tm,qbase,kvbase); commit((barrier_base+72+8*(0)));
      mma_qk(tm+128,qbase+bytes,kvbase); commit((barrier_base+72+8*(1))); commit((barrier_base+40+8*(0)));
      for (int n=0;n<blocks;++n) {
        int slot=(n&1)*2+1;
        wait((barrier_base+8+8*(slot)),(n/2)&1);
        wait((barrier_base+120+8*(0)),n&1); fence_after();
        mma_pv<0>(tm+256,tm+64,kvbase+slot*bytes,n!=0);
        wait((barrier_base+136+8*(0)),n&1); fence_after();
        mma_pv<1>(tm+256,tm+64,kvbase+slot*bytes,true); commit((barrier_base+152+8*(0)));
        if (n+1<blocks) {
          int ks=((n+1)&1)*2; wait((barrier_base+8+8*(ks)),((n+1)/2)&1); fence_after();
          mma_qk(tm,qbase,kvbase+ks*bytes); commit((barrier_base+72+8*(0)));
        }
        wait((barrier_base+120+8*(1)),n&1); fence_after();
        mma_pv<0>(tm+384,tm+192,kvbase+slot*bytes,n!=0);
        wait((barrier_base+136+8*(1)),n&1); fence_after();
        mma_pv<1>(tm+384,tm+192,kvbase+slot*bytes,true); commit((barrier_base+152+8*(1)));
        commit((barrier_base+40+8*(slot)));
        if(n+1<blocks) {
          int ks=((n+1)&1)*2;
          mma_qk(tm+128,qbase+bytes,kvbase+ks*bytes); commit((barrier_base+72+8*(1))); commit((barrier_base+40+8*(ks)));
        }
      }
    }
  } else if (warp < 8) {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(SOFTMAX_REGS) : "memory");
    unsigned stage = warp/4, addr = tm + stage*128 + ((warp%4)*32<<16);
    // TMEM addresses are warp-uniform; expose that once to ptxas.
    addr=__shfl_sync(0xffffffff,addr,0);
    int qb = qblock + stage;
    float qscale = 0.1275174305599474f * ((Scaled || p.qd) && qb < blocks ? __ldg(p.qd+head*blocks+qb) : 1.f);
    // Read-only per-head descale stream; avoid rebuilding the head offset.
    const float* key_scales=p.kd ? p.kd+head*blocks : nullptr;
    float sum=0, maximum=-INFINITY;
    for(int n=0;n<blocks;++n) {
      wait((barrier_base+72+8*(stage)),n&1); fence_after();
      float s[128];
      float m0=tmax64(addr,s), m1=tmax64(addr+64,s+64);
      float tilemax=fmaxf(m0,m1);
      if(n*128+128>p.s) {
        tilemax=-INFINITY;
#pragma unroll
        for(int i=0;i<128;++i) { if(n*128+i>=p.s) s[i]=-INFINITY; tilemax=fmaxf(tilemax,s[i]); }
      }
      float kd=(Scaled || p.kd) ? __ldg(key_scales+n) : 1.f;
      tilemax*=kd;
      float newmax=fmaxf(maximum,tilemax);
      if(n && (maximum-newmax)*qscale>=-4.f) newmax=maximum;
      float alpha=n ? exp2_fast((maximum-newmax)*qscale) : 0.f;
      maximum=newmax;
      fence_ld(); tstore1(addr,&alpha); fence_st(); fence_before(); arrive((barrier_base+88+8*(stage)));
      float multiplier=qscale*kd, bias=4.f-maximum*qscale;
#pragma unroll
      // Interleave independent score fragments; preserve the exact reduction tree.
      for(int j=0;j<32;j+=2) {
#pragma unroll
        for(int k=0;k<4;++k) { int i=j+k*32; fma2(s[i],s[i+1],multiplier,bias); s[i]=exp2_fast(s[i]); s[i+1]=exp2_fast(s[i+1]); }
      }
#pragma unroll
      for(int half=0;half<2;++half) {
        float packed[16];
#pragma unroll
        for(int i=0;i<16;++i) {
          int x=half*64+i*4;
          packed[i]=pack4(s[x],s[x+1],s[x+2],s[x+3]);
        }
        tstore16(addr+64+half*16,packed);

        fence_st(); fence_before(); arrive(shared_addr(half ? &b.p_tail[stage] : &b.p_ready[stage]));
      }

#pragma unroll
      for(int i=0;i<64;i+=2) add2(s[i],s[i+1],s[i+64],s[i+64+1]);
#pragma unroll
      for(int i=0;i<32;i+=2) add2(s[i],s[i+1],s[i+32],s[i+32+1]);
#pragma unroll
      for(int i=0;i<16;i+=2) add2(s[i],s[i+1],s[i+16],s[i+16+1]);
#pragma unroll
      for(int i=0;i<8;i+=2) add2(s[i],s[i+1],s[i+8],s[i+8+1]);
#pragma unroll
      for(int i=0;i<4;i+=2) add2(s[i],s[i+1],s[i+4],s[i+4+1]);
      add2(s[0],s[1],s[2],s[3]); s[0]+=s[1];

      sum=sum*alpha+s[0];
      wait((barrier_base+104+8*(stage)),n&1);
    }
    float stats[2]={sum,maximum*qscale}; tstore2(addr,stats); fence_st(); fence_before(); arrive((barrier_base+168+8*(stage)));
  } else {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(CORRECTION_REGS) : "memory");
    unsigned rows=(warp%4)*32<<16;
    // Hoist the warp row offset so later TMEM accesses share a uniform base.
    tm=__shfl_sync(0xffffffff,tm+rows,0); rows=0;
    for(int n=0;n<blocks;++n) {
#pragma unroll
      for(int stage=0;stage<2;++stage) {
        // One warp observes completion, then transfers ordering to its group.
        if(warp==8) { wait((barrier_base+88+8*(stage)),n&1); fence_after(); fence_before(); }
        asm volatile("barrier.sync 4, 128;" ::: "memory");
        fence_after();
        float alpha; tload1(tm+stage*128+rows,&alpha); fence_ld();
        // Cross-stage release staggers the two softmax groups after faster MMA issue.
        if(n==0) {
          if(stage==0) arrive((barrier_base+104+8*(0)));
        } else {
          arrive((barrier_base+104+8*(1-stage)));
        }
        if(n && __any_sync(0xffffffff,alpha<1.f)) {
#pragma unroll
          for(int x=0;x<4;++x) {
            float o[32]; tload32(tm+256+stage*128+x*32+rows,o); fence_ld();
#pragma unroll
            for(int i=0;i<32;++i) o[i]*=alpha;
            tstore32(tm+256+stage*128+x*32+rows,o);
          }
          fence_st();
        }
        fence_before(); arrive((barrier_base+120+8*(stage)));
      }
    }
    arrive((barrier_base+104+8*(1)));
    auto* output_smem=reinterpret_cast<uint32_t*>(sm+6*bytes);
#pragma unroll
    for(int stage=0;stage<2;++stage) {
      wait((barrier_base+152+8*(stage)),(blocks-1)&1); wait((barrier_base+168+8*(stage)),0); fence_after();
      float stats[2]; tload2(tm+stage*128+rows,stats); fence_ld();
      float inv=((Scaled || p.vd) ? __ldg(p.vd+head) : 1.f)/stats[0];
      int qr=(qblock+stage)*128+row;
      if(p.lse && qr<p.s) p.lse[head*p.s+qr]=(stats[1]+__log2f(stats[0])-4.f)*0.6931471805599453f;
#pragma unroll
      for(int half=0;half<4;++half) {
        float o[32]; tload32(tm+256+stage*128+half*32+rows,o); fence_ld();
#pragma unroll
        for(int i=0;i<16;i+=4) {
          unsigned ptr=shared_addr(output_smem+((row*64+half*16+i)^((row&7)*8)));
          unsigned v0=__float_as_uint(pack2(o[2*i]*inv,o[2*i+1]*inv));
          unsigned v1=__float_as_uint(pack2(o[2*i+2]*inv,o[2*i+3]*inv));
          unsigned v2=__float_as_uint(pack2(o[2*i+4]*inv,o[2*i+5]*inv));
          unsigned v3=__float_as_uint(pack2(o[2*i+6]*inv,o[2*i+7]*inv));
          asm volatile("st.shared.v4.b32 [%0],{%1,%2,%3,%4};" :: "r"(ptr),"r"(v0),"r"(v1),"r"(v2),"r"(v3) : "memory");
        }
      }
      asm volatile("barrier.sync 1, 128;" ::: "memory");
      for(int idx=row;idx<2048;idx+=128) {
        int r=idx/16, col=(idx%16)*4;
        int globalrow=(qblock+stage)*128+r;
        if(globalrow<p.s) {
          uint4 value=*reinterpret_cast<uint4*>(output_smem+((r*64+col)^((r&7)*8)));
          *reinterpret_cast<uint4*>(p.out+(int64_t(globalrow)*p.h+head)*128+col*2)=value;
        }
      }
      asm volatile("barrier.sync 1, 128;" ::: "memory");
    }
  }
  __syncthreads();
  if(warp==12) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(tm) : "memory");
}

// Host ABI is identical across precision-specific DSOs. Cross-dtype calls fail.
struct Plan { Params p; bool scaled; int smem; };
static thread_local std::string error;
bool make_map(CUtensorMap& map, void* ptr, int s, int h) {
  uint64_t dims[3]={128,uint64_t(h),uint64_t(s)};
  uint64_t strides[2]={128*1,uint64_t(h)*128*1};
  uint32_t box[3]={128,1,128}, steps[3]={1,1,1};
  CUresult result=cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_UINT8,
      3,ptr,dims,strides,box,steps,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if(result!=CUDA_SUCCESS) { const char* msg; cuGetErrorString(result,&msg); error=msg; return false; }
  return true;
}
void* create(int dtype,void* q,void* k,void* v,void* out,void* lse,int s,int h,
             const float* qd,const float* kd,const float* vd) {
  error.clear();
  if(dtype!=1) { error="fp8 library requires dtype=1"; return nullptr; }
  if(s<1 || h<1 || !q || !k || !v || !out || !lse) { error="invalid raw FA arguments"; return nullptr; }
  int dev; cudaGetDevice(&dev); cudaDeviceProp prop; cudaGetDeviceProperties(&prop,dev);
  if(prop.major!=10 || prop.minor!=3) { error="raw FA requires SM103"; return nullptr; }
  Plan* plan=new Plan{}; plan->scaled=qd && kd && vd; plan->smem=6*16384*1+32768+sizeof(Barriers);
  plan->p.out=static_cast<__nv_bfloat16*>(out);plan->p.lse=static_cast<float*>(lse);plan->p.s=s;plan->p.h=h;
  plan->p.qd=qd;plan->p.kd=kd;plan->p.vd=vd;
  if(!make_map(plan->p.q,q,s,h)||!make_map(plan->p.k,k,s,h)||!make_map(plan->p.v,v,s,h)) { delete plan; return nullptr; }
  cudaError_t e=plan->scaled ? cudaFuncSetAttribute(attention_fp8<true>,cudaFuncAttributeMaxDynamicSharedMemorySize,plan->smem) : cudaFuncSetAttribute(attention_fp8<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,plan->smem);
  if(e!=cudaSuccess) {error=cudaGetErrorString(e);delete plan;return nullptr;}
  return plan;
}
} // namespace RAW_VARIANT
extern "C" {
int fa_dtype() { return 1; }
void fa_debug(void* ptr, float* debug) { static_cast<RAW_VARIANT::Plan*>(ptr)->p.debug=debug; }
void fa_set_lse(void* ptr, float* lse) { static_cast<RAW_VARIANT::Plan*>(ptr)->p.lse=lse; }
const char* fa_error() { return RAW_VARIANT::error.c_str(); }
void* fa_create(int dtype,int,void* q,void* k,void* v,void* out,void* lse,void*,int s,int h,void*) {
  return RAW_VARIANT::create(dtype,q,k,v,out,lse,s,h,nullptr,nullptr,nullptr);
}
void* fa_create_scaled(int dtype,int,void* q,void* k,void* v,void* out,void* lse,void*,int s,int h,void*,const float* qd,const float* kd,const float* vd) {
  return RAW_VARIANT::create(dtype,q,k,v,out,lse,s,h,qd,kd,vd);
}
int fa_run(void* ptr,void* stream) {
  auto* p=static_cast<RAW_VARIANT::Plan*>(ptr);
  dim3 grid((p->p.s+255)/256,p->p.h);
  if(p->scaled) RAW_VARIANT::attention_fp8<true><<<grid,512,p->smem,static_cast<cudaStream_t>(stream)>>>(p->p);
  else RAW_VARIANT::attention_fp8<false><<<grid,512,p->smem,static_cast<cudaStream_t>(stream)>>>(p->p);
  cudaError_t e=cudaGetLastError();if(e!=cudaSuccess) RAW_VARIANT::error=cudaGetErrorString(e);return int(e);
}
int fa_smem(void* ptr) { return static_cast<RAW_VARIANT::Plan*>(ptr)->smem; }
void fa_destroy(void* ptr) { delete static_cast<RAW_VARIANT::Plan*>(ptr); }
}
