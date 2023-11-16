//----------------------------------------------------------------------------

#include "common.hpp"

//
// ... À COMPLÉTER (première étape uniquement) ...
//
// Réaliser la fonction qui sera utilisée par le thread qui assurera un
// dialogue en mode texte avec un client TCP.
// Le corps du traitement réalisé par ce thread doit être structuré comme
// ceci (en complétant notamment les opérations de communication avec le
// client) :
//
/***
    //---- prepare storage for a block of integer values ----
    auto storage=std::make_unique<int32_t []>(blockSize);
    int32_t *block=storage.get();

    //---- as long as some blocks are provided by the client ----
    for(PerfState state; state.next();)
    {
      //---- receive next block ----
      bool stop=false;
      for(int i=0; i<blockSize; ++i)
      {
        //---- each value is received in its textual form (11 chars) ----
        // Recevoir dans ``txtValue'' une chaîne devant contenir
        // exactement 11 caractères en provenance du client.
        // En cas de fin-de-fichier, il faudra la prendre en compte avec
        // ``stop=true;'' et quitter cette boucle avec  ``break;''.
        const std::string txtValue="... À REMPLACER ...";

        //---- convert this text into an integer value ----
        block[i]=int32_t(std::stol(txtValue));
      }
      if(stop)
      {
        break; // no more block to be received
      }

      //---- compute the sum of the integers in the block ----
      const int64_t sum=accumulateBlock(block);

      //---- convert this sum into its textual form (21 chars) ----
      const std::string txtSum=crs::txt("%%", std::setw(21), sum);

      //---- send this text to the client ----
      // Envoyer au client le texte préparé dans ``txtSum''.
    }
***/
//
// Bien entendu, cet algorithme doit être précédé et suivi de tout ce qui
// est nécessaire à la bonne gestion de la communication avec le client
// TCP.
//

// ...


//
// ... À COMPLÉTER (deuxième étape uniquement) ...
//
// Réaliser la fonction qui sera utilisée par le thread qui assurera un
// dialogue en mode binaire avec un client TCP.
// Le corps du traitement réalisé par ce thread doit être structuré comme
// ceci (en complétant notamment les opérations de communication avec le
// client) :
//
/***
    //---- prepare storage for a block of integer values ----
    auto storage=std::make_unique<int32_t []>(blockSize);
    int32_t *block=storage.get();
    const int blockBytes=int(blockSize*sizeof(int32_t));

    //---- as long as some blocks are provided by the client ----
    for(PerfState state; state.next();)
    {
      //---- receive next block (consider byte-order) ----
      // Recevoir dans le tableau ``block'' les ``blockBytes'' octets
      // en provenance du client.
      // Si les ``blockBytes'' octets attendus ne sont pas reçus,
      // alors il faut quitter cette boucle avec ``break;''.
      // nb : les ``blockSize'' entiers de 32 bits qui sont alors obtenus
      //      dans ``block'' sont dans l'ordre réseau ; ils ne sont donc pas
      //      directement exploitables.

      //---- compute the sum of the integers in the block ----
      int64_t sum=accumulateBlock(block);

      //---- send this sum to the client (consider byte-order) ----
      // Transmettre au client les octets constitutifs de la variable ``sum''.
      // nb : l'entier de 64 bits ``sum'' est dans l'ordre hôte ; il n'est
      //      donc pas directement transmissible.
    }
***/
//
// Bien entendu, cet algorithme doit être précédé et suivi de tout ce qui
// est nécessaire à la bonne gestion de la communication avec le client
// TCP.
//

// ...


int
main(int argc,
     char **argv)
{
  std::vector<std::string> args{argv, argv+argc};

  //---- check command line arguments ----
  if((crs::len(args)!=2)&&(crs::len(args)!=3))
  {
    crs::err("usage: % text_port [binary_port]\n", args[0]);
    crs::exit(1);
  }

  //---- extract local text port number ----
  uint16_t txtPortNumber=crs::len(args)>1
                         ? uint16_t(std::stoi(args[1])) : 0;
  SOCKET txtListenSocket=INVALID_SOCKET;

  //---- extract local binary port number ----
  uint16_t binPortNumber=crs::len(args)>2
                         ? uint16_t(std::stoi(args[2])) : 0;
  SOCKET binListenSocket=INVALID_SOCKET;

  //---- create text listen socket ----
  if(txtPortNumber!=0)
  {
    //
    // ... À COMPLÉTER (première étape uniquement) ...
    //
    // Préparer dans la variable ``txtListenSocket'' (qui existe déjà) une
    // socket TCP qui permettra d'attendre des connexions sur le port
    // ``txtPortNumber''.
    //

    // ...

    crs::out("host '%' waiting for text connections on port '%'...\n",
             crs::gethostname(), txtPortNumber);
  }

  //---- create binary listen socket ----
  if(binPortNumber!=0)
  {
    //
    // ... À COMPLÉTER (deuxième étape uniquement) ...
    //
    // Préparer dans la variable ``binListenSocket'' (qui existe déjà) une
    // socket TCP qui permettra d'attendre des connexions sur le port
    // ``binPortNumber''.
    //

    // ...

    crs::out("host '%' waiting for binary connections on port '%'...\n",
             crs::gethostname(), binPortNumber);
  }

  //
  // ... À COMPLÉTER (première étape uniquement) ...
  //
  // Faites en sorte que chaque nouvelle connexion TCP détectée sur
  // ``txtListenSocket'' provoque l'exécution dans un thread de la
  // fonction que vous avez écrite plus haut afin de réaliser le dialogue
  // TCP en mode texte concernant cette première étape.
  //
  // nb : cette partie du programme sera reformulée lorsque vous réaliserez
  //      l'étape suivante.
  //


  //
  // ... À COMPLÉTER (deuxième étape uniquement) ...
  //
  // Faites en sorte (en utilisant ``crs::select()'') que des nouvelles
  // connexions TCP puissent être détectées à la fois sur ``txtListenSocket''
  // et ``binListenSocket''.
  // * Dans le premier de ces deux cas il faudra provoquer l'exécution dans
  //   un thread de la fonction que vous avez écrite plus haut afin de
  //   réaliser le dialogue TCP en mode texte concernant la première étape.
  // * Dans le second de ces deux cas il faudra provoquer l'exécution dans
  //   un thread de la fonction que vous avez écrite plus haut afin de
  //   réaliser le dialogue TCP en mode binaire concernant cette deuxième
  //   étape.
  // 
  // nb : il vous faudra reformuler ce que vous avez fait à cette occasion
  //      lors de la première étape ; vous pouvez par exemple conserver votre
  //      code précédent en commentaire et le copier/coller ici afin de
  //      l'adapter à cette nouvelle situation.
  //


  //---- close listen sockets ----
  if(txtPortNumber!=0)
  {
    crs::close(txtListenSocket);
  }
  if(binPortNumber!=0)
  {
    crs::close(binListenSocket);
  }

  return 0;
}

//----------------------------------------------------------------------------
