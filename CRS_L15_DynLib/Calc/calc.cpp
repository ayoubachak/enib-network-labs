//----------------------------------------------------------------------------

#include "calc.hpp"

#include <iostream>
#include <sstream>

#if defined _WIN32
// trivial emulation of <dlfcn.h> features on Windows
# include <windows.h>
# define RTLD_LAZY 0
# define dlopen(name, flags) ((void *)LoadLibrary((name)))
# define dlsym(lib, sym)     ((void *)GetProcAddress((HMODULE)(lib), (sym)))
# define dlclose(lib)        (!FreeLibrary((HMODULE)(lib)))
# define dlerror()           ("dynamic library error")
#else
# include <dlfcn.h>
#endif

namespace calc {

static
std::string
compose_library_name(const std::string &operation)
{
  const auto lib_name="Calc"+operation; // application-specific
  const auto system_specific_lib_name=
#if defined _WIN32
    lib_name+".dll";
#elif defined __APPLE__
    "lib"+lib_name+".dylib";
#else
    "lib"+lib_name+".so";
#endif
  const auto &sub_dir=lib_name; // application-specific
  return sub_dir+"/"+system_specific_lib_name;
}

static
std::string
compose_function_name(const std::string &operation)
{
  return operation+"_operation"; // application-specific
}

static
bool // go on
apply_operation(std::vector<double> &values,
                const std::string &operation)
{
  bool go_on=true;
  std::string lib_name=compose_library_name(operation);
  std::string fnct_name=compose_function_name(operation);

  //
  // ... À COMPLÉTER ...
  //
  // Charger la bibliothèque dynamique dont le nom est donné par la
  //   variable «lib_name».
  // Y retrouver la fonction dont le nom est donné par la
  //   variable «fnct_name» afin de l'appeler.
  // Cette fonction est de type «OperationFunction» (voir calc.hpp)
  //   c'est à dire qu'elle attend en paramètre une référence sur
  //   un tableau dynamique de réels («values») et renvoie un résultat
  //   booléen («go_on»).
  // Une fois l'appel réalisé, il faudra libérer la bibliothèque
  //   dynamique précédemment chargée.
  //

  // ...

  return go_on;
}

std::string
prompt(const std::vector<double> &values)
{
  std::ostringstream output;
  output << "stack:";
  for(const auto &value: values)
  {
    output << ' ' << value;
  }
  output << "\n? ";
  return output.str();
}

bool // go on
input(std::vector<double> &values,
      const std::string &input_line)
{
  std::istringstream input{input_line};
  for(std::string word; input >> word; )
  {
    try
    {
      values.emplace_back(std::stod(word));
      continue;
    }
    catch(...)
    {
      // not a real value, then consider it is an operation
    }
    if(!apply_operation(values, word))
    {
      return false;
    }
  }
  return true;
}

} // namespace calc

//----------------------------------------------------------------------------
