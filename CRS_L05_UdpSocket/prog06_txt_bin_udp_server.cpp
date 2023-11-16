//----------------------------------------------------------------------------

#include "crsUtils.hpp"

int
main(int argc,
     char **argv)
{
  std::vector<std::string> args{argv, argv+argc};

  //---- check command line arguments ----
  if(crs::len(args)!=3)
  {
    crs::err("usage: % txt_port bin_port\n", args[0]);
    crs::exit(1);
  }

  //---- extract local port numbers ----
  uint16_t txtPortNumber=uint16_t(std::stoi(args[1]));
  uint16_t binPortNumber=uint16_t(std::stoi(args[2]));

  //---- create UDP socket (text) ----
  //
  // ... À COMPLÉTER {1} ...
  //
  // (semblable à prog02_txt_udp_server.cpp)
  //

  // ...

  //---- create UDP socket (binary) ----
  //
  // ... À COMPLÉTER {2} ...
  //
  // (semblable à prog04_bin_udp_server.cpp)
  //

  // ...

  crs::out("host '%' waiting for text/binary messages on ports '%/%'...\n",
           crs::gethostname(), txtPortNumber, binPortNumber);
  for(;;) // as long as dialog can go on...
  {
    //---- wait for incoming information ----
    //
    // ... À COMPLÉTER {4} ...
    //
    // Surveiller simultanément (avec ``crs::select()'') les messages arrivant
    // sur les deux sockets.
    //

    // ...

    //---- react to textual request (if available) ----
    //
    // ... À COMPLÉTER {5} ...
    //
    // Si la socket pour les échanges textuels est prête à fournir des données
    // (``voir crs::find()''), effectuer l'échange textuel requête/réponse.
    //
    // (semblable à prog02_txt_udp_server.cpp)
    //
    if( false ) // ... MODIFIER CETTE CONDITION ...
    {

    }
    // ...

    //---- react to binary request (if available) ----
    //
    // ... À COMPLÉTER {6} ...
    //
    // Si la socket pour les échanges binaires est prête à fournir des données
    // (``voir crs::find()''), effectuer l'échange binaire requête/réponse.
    //
    // (semblable à prog04_bin_udp_server.cpp)
    //
    if( false ) // ... MODIFIER CETTE CONDITION ...
    {

    }
    // ...
  }

  //---- close UDP sockets ----
  //
  // ... À COMPLÉTER {3} ...
  //
  // (semblable à prog02_txt_udp_server.cpp et prog04_bin_udp_server.cpp)
  //

  // ...

  return 0;
}

//----------------------------------------------------------------------------
