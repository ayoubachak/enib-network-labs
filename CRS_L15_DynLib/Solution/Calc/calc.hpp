//----------------------------------------------------------------------------

#ifndef CALC_HPP
#define CALC_HPP

#include <string>
#include <vector>

namespace calc {

using OperationFunction = bool (*)(std::vector<double> &);

std::string
prompt(const std::vector<double> &values);

bool // go on
input(std::vector<double> &values,
      const std::string &input_line);

} // namespace calc

#endif // CALC_HPP

//----------------------------------------------------------------------------
