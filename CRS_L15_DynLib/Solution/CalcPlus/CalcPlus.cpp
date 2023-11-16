//----------------------------------------------------------------------------

#include <iostream>
#include <vector>

extern "C"
bool // go on
Plus_operation(std::vector<double> &values)
{
  if(size(values)<2)
  {
    std::cerr << "2 values expected\n";
  }
  else
  {
    const auto a=values.back();
    values.pop_back();
    const auto b=values.back();
    values.pop_back();
    values.emplace_back(a+b);
  }
  return true;
}

//----------------------------------------------------------------------------
