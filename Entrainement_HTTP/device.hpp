//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#ifndef DEVICE_HPP
#define DEVICE_HPP

#include "crsUtils.hpp"

namespace device {

inline struct
{
  std::atomic_flag lock;
  real64_t clock;
  uint32_t counter;
} device_={ATOMIC_FLAG_INIT, crs::gettimeofday(), 0};

inline
void
reset()
{
  while(device_.lock.test_and_set(std::memory_order_acquire)) { }
    device_.clock=crs::gettimeofday();
    device_.counter=0;
  device_.lock.clear(std::memory_order_release);
}

inline
real64_t
clock()
{
  while(device_.lock.test_and_set(std::memory_order_acquire)) { }
    const auto clock=crs::gettimeofday()-device_.clock;
  device_.lock.clear(std::memory_order_release);
  return clock;
}

inline
uint32_t
counter()
{
  while(device_.lock.test_and_set(std::memory_order_acquire)) { }
    const auto counter=++device_.counter;
  device_.lock.clear(std::memory_order_release);
  return counter;
}

} //namespace device

#endif // DEVICE_HPP

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
