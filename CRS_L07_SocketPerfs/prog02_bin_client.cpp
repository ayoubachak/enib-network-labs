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
  const int blockBytes=int(blockSize*sizeof(int32_t));

  //
  // ... À COMPLÉTER ...
  //
  // Réaliser un client TCP qui contacte le serveur désigné par les
  // arguments de la ligne de commande, exactement comme dans le cas du
  // programme ``prog01_txt_client.cpp''.
  // Il s'agit désormais de remplacer l'utilisation précédente de messages
  // textuels par des transferts directs des blocs de données dans la
  // connexion TCP.
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

    //---- compute the sum of the integers in the block ----
    const int64_t sum=accumulateBlock(block);

    //---- send next block (consider byte-order) ----
    // Transmettre au serveur les octets les ``blockBytes'' octets
    // constitutifs du tableau ``block''.
    // nb : les ``blockSize'' entiers de 32 bits préparés dans ``block''
    //      sont dans l'ordre hôte ; ils ne sont donc pas directement
    //      transmissibles.

    //---- obtain reply from server (consider byte-order) ----
    int64_t reply;
    // Recevoir depuis le client les octets constitutifs de la variable
    // ``reply''.
    // nb : l'entier de 64 bits ``reply'' qui est alors obtenu est dans
    //      l'ordre réseau ; il n'est donc pas directement exploitable.

    //---- check reply ----
    if(sum!=reply)
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

  // ...

  return 0;
}

//----------------------------------------------------------------------------
