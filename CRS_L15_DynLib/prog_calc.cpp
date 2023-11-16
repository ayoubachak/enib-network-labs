//----------------------------------------------------------------------------

#include "calc.hpp"
#include <iostream>

int
main()
{
  std::vector<double> values;
  for(;;)
  {
    std::cout << calc::prompt(values) << std::flush;
    std::string line;
    if(!std::getline(std::cin, line))
    {
      std::cerr << "<EOF>\n";
      break;
    }
    if(!calc::input(values, line))
    {
      break;
    }
  }
  return 0;
}

//----------------------------------------------------------------------------
