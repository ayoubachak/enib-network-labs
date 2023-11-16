//----------------------------------------------------------------------------

#ifndef COMMON_HPP
#define COMMON_HPP 1

#include "crsUtils.hpp"
#include <random>

constexpr int blockSize=16*1024;

inline
void
generateBlock(int32_t *block)
{
  static std::default_random_engine rndGen{std::random_device{}()};
  static std::uniform_int_distribution<int32_t> uniDist{0, 10};
  std::iota(block, block+blockSize, uniDist(rndGen));
}

inline
int64_t
accumulateBlock(const int32_t *block)
{
  return std::accumulate(block, block+blockSize, int64_t{});
}

class PerfState
{
public:

  PerfState(int iterCount=-1)
  : blockCount_{0}
  , iterCount_{iterCount}
  , t0_{crs::gettimeofday()}
  , e0_{crs::cpuEnergy()}
  {
#if !defined NDEBUG
    crs::err("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
             "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
    crs::err("  Built for debug purpose at first "
             "(measurements will not be relevant).\n");
    crs::err("  For an actual experiment, rebuild with: "
             "   make rebuild opt=1\n");
    crs::err("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
             "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
#endif
    if(iterCount_==0)
    {
      crs::out("running during at least % s...\n", minDuration_);
    }
    else if(iterCount_>0)
    {
      crs::out("running % iterations...\n", iterCount_);
    }
    else
    {
      crs::out("starting iterations...\n");
    }
  }

  ~PerfState()
  {
    const auto duration=crs::gettimeofday()-t0_;
    const double energy=crs::cpuEnergy()-e0_;
    crs::out("% blocks in % s (% block/s, % Joules)\n",
             blockCount_-1, duration, (blockCount_-1)/duration, energy);
  }

  bool // go on
  next()
  {
    ++blockCount_;
    if(iterCount_>0)
    {
      return blockCount_<=iterCount_;
    }
    if(iterCount_==0)
    {
      const auto duration=crs::gettimeofday()-t0_;
      return duration<minDuration_;
    }
    return true;
  }

private:
  static constexpr double minDuration_=5.0;
  int blockCount_;
  const int iterCount_;
  const double t0_;
  const double e0_;
};

#endif // COMMON_HPP

//----------------------------------------------------------------------------
