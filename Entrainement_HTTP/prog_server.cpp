//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#include "device.hpp"

// ...

int
main(int argc,
     char **argv)
{
  //---- check command line arguments ----
  const auto args=std::vector<std::string>{argv, argv+argc};
  if(crs::len(args)!=2)
  {
    crs::err("usage: % port_number\n", args[0]);
    crs::exit(1);
  }
  crs::out("from command line:\n");
  const auto port_number=uint16_t(std::stoi(args[1]));
  crs::out("• port number: %\n", port_number);

  // ...

  return 0;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
