//----------------------------------------------------------------------------

#include "crsUtils.hpp"

void
txtDialogThread(SOCKET dialogSocket)
{
  try
  {
    for(;;) // as long as dialog can go on...
    {
      //
      // ... À COMPLÉTER {7} ...
      //
      // Reprendre ici ce qui a été réalisé dans la fonction ``dialogThread()''
      // de prog03_txt_tcp_server.cpp
      //

      // ...
    }
  }
  catch(const std::exception &e)
  {
    crs::err("\n!!! Exception: % !!!\n", e.what());
  }
  catch(...)
  {
    crs::err("\n!!! Unknown exception !!!\n");
  }

  //---- close dialog socket in any case! ----
  crs::close(dialogSocket);

  crs::out("text client disconnected\n");
}

void
binDialogThread(SOCKET dialogSocket)
{
  try
  {
    for(;;) // as long as dialog can go on...
    {
      //
      // ... À COMPLÉTER {8} ...
      //
      // Reprendre ici ce qui a été réalisé dans la fonction ``dialogThread()''
      // de prog04_bin_tcp_server.cpp
      //

      // ...
    }
  }
  catch(const std::exception &e)
  {
    crs::err("\n!!! Exception: % !!!\n", e.what());
  }
  catch(...)
  {
    crs::err("\n!!! Unknown exception !!!\n");
  }

  //---- close dialog socket in any case! ----
  crs::close(dialogSocket);

  crs::out("binary client disconnected\n");
}

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

  //---- create listen socket (text) ----
  //
  // ... À COMPLÉTER {1} ...
  //
  // (semblable à prog03_txt_tcp_server.cpp)
  //

  // ...

  //---- create listen socket (binary) ----
  //
  // ... À COMPLÉTER {2} ...
  //
  // (semblable à prog04_bin_tcp_server.cpp)
  //

  // ...

  crs::out("host '%' waiting for text/binary connections on ports '%/%'...\n",
           crs::gethostname(), txtPortNumber, binPortNumber);
  for(;;)
  {
    //---- wait for connections ----
    //
    // ... À COMPLÉTER {4} ...
    //
    // Surveiller simultanément (avec ``crs::select()'') les demandes de
    // connexion sur les deux sockets d'écoute.
    //

    // ...

    //---- react to text connection (if available) ----
    //
    // ... À COMPLÉTER {5} ...
    //
    // Si la socket d'écoute pour les échanges textuels a reçu une nouvelle
    // demande de connexion (``voir crs::find()''), accepter cette connexion
    // et démarrer un thread qui exécutera le dialogue avec le client dans
    // la fonction ``txtDialogThread()''.
    //
    // (semblable à prog03_txt_tcp_server.cpp)
    //
    if( false ) // ... MODIFIER CETTE CONDITION ...
    {

    }
    // ...

    //---- react to binary connection (if available) ----
    //
    // ... À COMPLÉTER {6} ...
    //
    // Si la socket d'écoute pour les échanges binaires a reçu une nouvelle
    // demande de connexion (``voir crs::find()''), accepter cette connexion
    // et démarrer un thread qui exécutera le dialogue avec le client dans
    // la fonction ``binDialogThread()''.
    //
    // (semblable à prog04_bin_tcp_server.cpp)
    //
    if( false ) // ... MODIFIER CETTE CONDITION ...
    {

    }
    // ...
  }

  //---- close listen sockets ----
  //
  // ... À COMPLÉTER {3} ...
  //
  // (semblable à prog03_txt_tcp_server.cpp et prog04_bin_tcp_server.cpp)
  //

  // ...

  return 0;
}

//----------------------------------------------------------------------------
