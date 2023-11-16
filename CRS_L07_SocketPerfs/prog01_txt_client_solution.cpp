//----------------------------------------------------------------------------

#include "common.hpp"

int
main(int argc,
     char **argv)
{
  std::vector<std::string> args{argv, argv+argc};

  //---- check command line arguments ----
  if((crs::len(args)!=3)&&(crs::len(args)!=4))
  {
    crs::err("usage: % destination port [iteration_count]\n", args[0]);
    crs::exit(1);
  }
  const int iterCount=crs::len(args)==4 ? std::stoi(args[3]) : 0;

  //---- prepare storage for a block of integer values ----
  auto storage=std::make_unique<int32_t []>(blockSize);
  int32_t *block=storage.get();

  //---- extract destination IP address ----
  uint32_t ipAddress;
  //
  // ... À COMPLÉTER ...
  //
  // Déterminer dans ``ipAddress'' l'adresse IP de ``args[1]''.
  //
  ipAddress=crs::gethostbyname(args[1]);
  // ...

  //---- extract destination port number ----
  uint16_t portNumber=uint16_t(std::stoi(args[2]));

  //
  // ... À COMPLÉTER ...
  //
  // Réaliser un client TCP qui contacte le serveur désigné par
  // ``ipAddress'' et ``portNumber'' pour lui transmettre les blocs
  // de données 
  // Le corps du traitement réalisé par ce client doit être structuré
  // comme ceci (en complétant notamment les opérations de
  // communication avec le serveur) :
  //
  /***
  //---- repeat as long as this experiment should last ----
  for(PerfState state{iterCount}; state.next(); )
  {
    //---- prepare next block of integer values ----
    generateBlock(block);

    //---- send next block ----
    for(int i=0; i<blockSize; ++i)
    {
      //---- convert each value into its textual form (11 chars) ----
      std::string txtValue=crs::txt("%%", std::setw(11), block[i]);

      //---- send this text to the server ----
      // Envoyer au serveur le texte préparé dans ``txtValues''.
    }

    //---- compute the sum of the integers in the block ----
    const int64_t sum=accumulateBlock(block);

    //---- convert this sum into its textual form (21 chars) ----
    const std::string txtSum=crs::txt("%%", std::setw(21), sum);

    //---- obtain reply from server ----
    // Recevoir dans ``reply'' une chaîne devant contenir
    // exactement 21 caractères en provenance du serveur.
    //
    const std::string reply="... À REMPLACER ...";

    //---- check reply ----
    if(txtSum!=reply)
    {
      throw std::runtime_error{"mismatch!"};
    }
  }
  ***/
  //
  // Bien entendu, cet algorithme doit être précédé et suivi de tout ce qui
  // est nécessaire à la bonne gestion de la communication avec le serveur
  // TCP.
  //

  //---- create client socket ----
  SOCKET clientSocket=crs::socket(PF_INET, SOCK_STREAM, 0);
  // ... connected to the specified destination/port
  crs::connect(clientSocket, ipAddress, portNumber);

  //---- repeat as long as this experiment should last ----
  for(PerfState state{iterCount}; state.next(); )
  {
    //---- prepare next block of integer values ----
    generateBlock(block);

    //---- send next block ----
    for(int i=0; i<blockSize; ++i)
    {
      //---- convert each value into its textual form (11 chars) ----
      std::string txtValue=crs::txt("%%", std::setw(11), block[i]);

      //---- send this text to the server ----
      crs::sendAll(clientSocket, txtValue);
    }

    //---- compute the sum of the integers in the block ----
    const int64_t sum=accumulateBlock(block);

    //---- convert this sum into its textual form (21 chars) ----
    const std::string txtSum=crs::txt("%%", std::setw(21), sum);

    //---- obtain reply from server ----
    const std::string reply=crs::recvAll(clientSocket , 21);

    //---- check reply ----
    if(txtSum!=reply)
    {
      throw std::runtime_error{"mismatch!"};
    }
  }

  //---- close client socket ----
  crs::close(clientSocket);

  // ...

  return 0;
}

//----------------------------------------------------------------------------
