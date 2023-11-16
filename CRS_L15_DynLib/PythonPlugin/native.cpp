//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>
#include <string>
#include <memory>

//~~ step 01 ~~ no argument/result ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

extern "C"
void
minimal()
{
  std::cout << "native function: " << __func__ << "()\n";
}

//~~ step 02 ~~ simple types as arguments/result ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

extern "C"
double
simple_types(int n,
             float v,
             const char *s)
{
  std::cout << "native function: " << __func__ << "()\n";
  std::cout << "  n="  << n << "  v="  << v << "  s="  << s << '\n';
  return double(n)+v;
}

//~~ step 03 ~~ access by reference ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

extern "C"
void
references(int &inout_count,
           double &inout_value)
{
  std::cout << "native function: " << __func__ << "()\n";
  std::cout << "  " << inout_count << " and " << inout_value;
  inout_count+=5;
  inout_value*=10.0;
  std::cout << "    are now turned into    "
            << inout_count << " and " << inout_value << '\n';
}

// ~~ step 04 ~~ access array ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

extern "C"
double
access_array(double *values,
             int count)
{
  std::cout << "native function: " << __func__ << "()\n";
  std::sort(values, values+count);
  return std::accumulate(values, values+count, 0.0);
}

//~~ step 05 ~~ create/use/destroy something ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Something
{
public:

  Something(int count)
  : name_{"here are "+std::to_string(count)+" generated values"}
  , values_(count)
  {
    std::cout << "... creating Something @" << this << '\n';
    const auto div=count>1 ? 1.0/(count-1) : 1.0;
    std::generate(begin(values_), end(values_),
      [&, i=0]() mutable
      {
        return (i++)*div;
      });
  }

  ~Something()
  {
    std::cout << "... destroying Something @" << this << '\n';
  }

  const char *
  c_name() const
  {
    return data(name_);
  }

  const double *
  values() const
  {
    return data(values_);
  }

  int
  value_count() const
  {
    return int(size(values_));
  }

private:
  std::string name_{};
  std::vector<double> values_{};
};

extern "C"
Something *
create_Something(int count)
{
  std::cout << "native function: " << __func__ << "()\n";
  // dynamically allocate
  auto something=std::make_unique<Something>(count);
  // release ownership to prevent automatic deletion
  return something.release();
}

extern "C"
void
use_Something(const Something &something,
              const char * &out_c_name,
              const double * &out_values,
              int &out_value_count)
{
  std::cout << "native function: " << __func__ << "()\n";
  out_c_name=something.c_name();
  out_values=something.values();
  out_value_count=something.value_count();
}

extern "C"
void
destroy_Something(Something &something)
{
  std::cout << "native function: " << __func__ << "()\n";
  // take ownership back to ensure automatic deletion
  const auto ptr=std::unique_ptr<Something>{&something};
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
