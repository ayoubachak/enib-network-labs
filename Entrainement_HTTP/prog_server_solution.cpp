//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#include "device.hpp"

struct RequestHeader
{
  bool failure{};
  std::string request_method{};
  std::string request_uri{};
  std::string connection{};
  std::string content_length{};
  std::string upgrade{};
  std::string ws_key{};
  int length{};
};

RequestHeader
read_request_header(SOCKET dialog_socket)
{
  auto rh=RequestHeader{};
  for(;;)
  {
    const auto l=crs::recvLine(dialog_socket);
    crs::out("http-request header: %", l);
    if(empty(l)||(l=="\n")||(l=="\r\n"))
    {
      break; // end of header
    }
    if(empty(rh.request_method)) // first line
    {
      crs::extract(l, rh.request_method, rh.request_uri);
    }
    else if(crs::extract(l, "Connection:", rh.connection)==2)
    {
      // nothing more to be done
    }
    else if(crs::extract(l, "Content-Length:", rh.content_length)==2)
    {
      // nothing more to be done
    }
    else if(crs::extract(l, "Upgrade:", rh.upgrade)==2)
    {
      // nothing more to be done
    }
    else if(crs::extract(l, "Sec-WebSocket-Key:", rh.ws_key)==2)
    {
      // nothing more to be done
    }
  }
  if(empty(rh.request_method))
  {
    rh.failure=true; // no request
  }
  else
  {
    rh.connection=crs::find(rh.connection, "close")!=-1
      ? "close" : "keep-alive"; // assume keep-alive by default
    if(!empty(rh.content_length))
    {
      rh.length=std::stoi(rh.content_length);
    }
  }
  return rh;
}

bool // keep-alive
handle_file(SOCKET dialog_socket,
            const RequestHeader &rh)
{
  const auto path=std::string{"client_solution.html"};
  crs::out("--> sending %\n", path);
  const auto header=crs::txt(
    "HTTP/1.1 200 OK\r\n"
    "Connection: %\r\n"
    "Content-Type: text/html\r\n"
    "Content-Length: %\r\n"
    "\r\n",
    rh.connection,
    crs::fileSize(path));
  crs::sendAll(dialog_socket, header);
  char buffer[0x400];
  int input=crs::openR(path);
  for(;;)
  {
    int r=crs::read(input, buffer, sizeof(buffer));
    if(r==0)
    {
      break; // EOF
    }
    crs::sendAll(dialog_socket, buffer, r);
  }
  crs::close(input);
  return rh.connection!="close";
}

bool // keep-alive
handle_txt(SOCKET dialog_socket,
           const RequestHeader &rh)
{
  const auto content=crs::recvAll(dialog_socket, rh.length);
  if(empty(content))
  {
    crs::out("<EOF>\n");
    return false;
  }
  if(crs::len(content)!=rh.length)
  {
    crs::err("% bytes expected\n", rh.length);
    return false;
  }
  crs::out("txt request: %\n", content);
  auto reply=std::string{"???"};
  if(content=="reset")
  {
    device::reset();
    reply="OK";
  }
  else if(content=="clock")
  {
    reply=crs::txt("%", device::clock());
  }
  else if(content=="counter")
  {
    reply=crs::txt("%", device::counter());
  }
  else if(content=="all")
  {
    reply=crs::txt("% %", device::clock(), device::counter());
  }
  crs::out("txt reply: %\n", reply);
  const auto header=crs::txt(
    "HTTP/1.1 200 OK\r\n"
    "Connection: %\r\n"
    "Content-Type: text/plain\r\n"
    "Content-Length: %\r\n"
    "\r\n",
    rh.connection,
    crs::len(reply));
  crs::sendAll(dialog_socket, header);
  crs::sendAll(dialog_socket, reply);
  return rh.connection!="close";
}

bool // keep-alive
handle_bin(SOCKET dialog_socket,
           const RequestHeader &rh)
{
  int16_t tmp_request;
  if(rh.length!=sizeof(tmp_request))
  {
    throw std::runtime_error{"bad Content-Length for request"};
  }
  const auto r=crs::recvAll(
    dialog_socket, &tmp_request, sizeof(tmp_request));
  if(r==0)
  {
    crs::out("<EOF>\n");
    return false;
  }
  if(r!=sizeof(tmp_request))
  {
    crs::err("% bytes expected\n", sizeof(tmp_request));
    return false;
  }
  const auto send_header=
    [&](const auto &bytes)
    {
      const auto header=crs::txt(
        "HTTP/1.1 200 OK\r\n"
        "Connection: %\r\n"
        "Content-Type: application/octet-stream\r\n"
        "Content-Length: %\r\n"
        "\r\n",
        rh.connection,
        bytes);
      crs::sendAll(dialog_socket, header);
    };
  const auto send_binary=
    [&](const auto &value, const auto &hton)
    {
      const auto tmp=hton(value);
      crs::sendAll(dialog_socket, &tmp, sizeof(tmp));
    };
  const auto request=crs::ntoh_i16(tmp_request);
  crs::out("bin request: %\n", request);
  if(request==0)
  {
    const auto reply=request;
    device::reset();
    crs::out("bin reply: %\n", reply);
    send_header(sizeof(reply));
    send_binary(reply, crs::hton_i16);
  }
  else if(request==1)
  {
    const auto reply=request;
    const auto clock=device::clock();
    crs::out("bin reply: % %\n", reply, clock);
    send_header(sizeof(reply)+sizeof(clock));
    send_binary(reply, crs::hton_i16);
    send_binary(clock, crs::hton_r64);
  }
  else if(request==2)
  {
    const auto reply=request;
    const auto counter=device::counter();
    crs::out("bin reply: % %\n", reply, counter);
    send_header(sizeof(reply)+sizeof(counter));
    send_binary(reply, crs::hton_i16);
    send_binary(counter, crs::hton_ui32);
  }
  else if(request==3)
  {
    const auto reply=request;
    const auto clock=device::clock();
    const auto counter=device::counter();
    crs::out("bin reply: % % %\n", reply, clock, counter);
    send_header(sizeof(reply)+sizeof(clock)+sizeof(counter));
    send_binary(reply, crs::hton_i16);
    send_binary(clock, crs::hton_r64);
    send_binary(counter, crs::hton_ui32);
  }
  else
  {
    const auto reply=int16_t{-1};
    crs::out("bin reply: %\n", reply);
    send_header(sizeof(reply));
    send_binary(reply, crs::hton_i16);
  }
  return rh.connection!="close";
}

void
handle_websocket_txt(SOCKET dialog_socket,
                     std::string request)
{
  crs::out("txt request: %\n", request);
  std::string reply="???";
  if(request=="reset")
  {
    device::reset();
    reply="OK";
  }
  else if(request=="clock")
  {
    reply=crs::txt("%", device::clock());
  }
  else if(request=="counter")
  {
    reply=crs::txt("%", device::counter());
  }
  else if(request=="all")
  {
    reply=crs::txt("% %", device::clock(), device::counter());
  }
  crs::out("txt reply: %\n", reply);
  crs::wsSend(dialog_socket, reply);
}

void
handle_websocket_bin(SOCKET dialog_socket,
                     char *request_buffer,
                     int length)
{
  auto tmp_request=int16_t{};
  if(length!=sizeof(tmp_request))
  {
    crs::err("% bytes expected\n", sizeof(tmp_request));
    return;
  }
  crs::unpack_bytes(request_buffer, length, tmp_request);
  const auto request=crs::ntoh_i16(tmp_request);
  crs::out("bin request: %\n", request);
  char buffer[100];
  auto amount=0;
  if(request==0)
  {
    const auto reply=request;
    device::reset();
    crs::out("bin reply: %\n", reply);
    amount=crs::pack_bytes(buffer, sizeof(buffer),
                           crs::hton_i16(reply));
  }
  else if(request==1)
  {
    const auto reply=request;
    const auto clock=device::clock();
    crs::out("bin reply: % %\n", reply, clock);
    amount=crs::pack_bytes(buffer, sizeof(buffer),
                           crs::hton_i16(reply),
                           crs::hton_r64(clock));
  }
  else if(request==2)
  {
    const auto reply=request;
    const auto counter=device::counter();
    crs::out("bin reply: % %\n", reply, counter);
    amount=crs::pack_bytes(buffer, sizeof(buffer),
                           crs::hton_i16(reply),
                           crs::hton_ui32(counter));
  }
  else if(request==3)
  {
    const auto reply=request;
    const auto clock=device::clock();
    const auto counter=device::counter();
    crs::out("bin reply: % % %\n", reply, clock, counter);
    amount=crs::pack_bytes(buffer, sizeof(buffer),
                           crs::hton_i16(reply),
                           crs::hton_r64(clock),
                           crs::hton_ui32(counter));
  }
  else
  {
    const auto reply=int16_t(-1);
    crs::out("bin reply: %\n", reply);
    amount=crs::pack_bytes(buffer, sizeof(buffer),
                           crs::hton_i16(reply));
  }
  crs::wsSend(dialog_socket, buffer, amount, crs::WS_BIN);
}

bool // keep-alive
handle_websocket(SOCKET dialog_socket,
                 const RequestHeader &rh)
{
  crs::out("--> upgrading to websocket\n");
  const auto header=crs::txt(
    "HTTP/1.1 101 Switching Protocols\r\n"
    "Connection: Upgrade\r\n"
    "Upgrade: websocket\r\n"
    "Sec-WebSocket-Accept: %\r\n"
    "\r\n",
    crs::wsHandshake(rh.ws_key));
  crs::sendAll(dialog_socket, header);
  for(;;)
  {
    char buffer[100];
    const auto [opcode, length]=
      crs::wsRecv(dialog_socket, buffer, sizeof(buffer));
    if(length==0)
    {
      break; // nothing more available
    }
    if(opcode==crs::WS_TXT)
    {
      handle_websocket_txt(dialog_socket, std::string{buffer});
    }
    else if(opcode==crs::WS_BIN)
    {
      handle_websocket_bin(dialog_socket, buffer, length);
    }
    else if(opcode==crs::WS_PING)
    {
      crs::out("ws_ping\n");
      crs::wsSend(dialog_socket, std::string{buffer}, crs::WS_PONG);
    }
    else if(opcode==crs::WS_CLOSE)
    {
      crs::out("ws_close\n");
      crs::wsSend(dialog_socket, std::string{buffer}, crs::WS_CLOSE);
      break; // stop websocket dialog
    }
  }
  return false; // close connection after websocket
}

bool // keep-alive
handle_error(SOCKET dialog_socket,
             const RequestHeader &rh)
{
  crs::out("--> sending 404 Not Found: % %\n",
           rh.request_method, rh.request_uri);
  const auto content=crs::txt(
    "<!DOCTYPE html>\n"
    "<html><head>\n"
    "<meta charset=\"utf-8\">\n"
    "</head><body>\n"
    "<h2>404 - Not Found</h2>\n"
    "<p>[<a href=\"/\">home</a>]</p>\n"
    "<hr>\n"
    "<p><i>method:</i> <b>%</b></p>\n"
    "<p><i>uri:</i> <b>%</b></p>\n"
    "<hr>\n"
    "</body></html>\n",
    rh.request_method,
    rh.request_uri);
  const auto header=crs::txt(
    "HTTP/1.1 404 Not Found\r\n"
    "Connection: %\r\n"
    "Content-Type: text/html\r\n"
    "Content-Length: %\r\n"
    "\r\n",
    rh.connection,
    crs::len(content));
  crs::sendAll(dialog_socket, header);
  crs::sendAll(dialog_socket, content);
  return rh.connection!="close";
}

void
dialog_thread(SOCKET dialog_socket)
{
  try
  {
    for(;;) // as long as dialog can go on...
    {
      const auto rh=read_request_header(dialog_socket);
      if(rh.failure)
      {
        break;
      }
      else if((rh.request_method=="GET")&&
              (rh.request_uri=="/"))
      {
        if(!handle_file(dialog_socket, rh))
        {
          break; // dialog stops here
        }
      }
      else if((rh.request_method=="POST")&&
              (rh.request_uri=="/txt"))
      {
        if(!handle_txt(dialog_socket, rh))
        {
          break; // dialog stops here
        }
      }
      else if((rh.request_method=="POST")&&
              (rh.request_uri=="/bin"))
      {
        if(!handle_bin(dialog_socket, rh))
        {
          break; // dialog stops here
        }
      }
      else if((rh.request_method=="GET")&&
              (rh.request_uri=="/ws")&&
              (rh.upgrade=="websocket"))
      {
        if(!handle_websocket(dialog_socket, rh))
        {
          break; // dialog stops here
        }
      }
      else // the last resort!
      {
        if(!handle_error(dialog_socket, rh))
        {
          break; // dialog stops here
        }
      }
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
  crs::close(dialog_socket);
  crs::out("client disconnected\n");
}

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

  const auto listen_socket=crs::socket(PF_INET, SOCK_STREAM, 0);
  crs::setReuseAddrOption(listen_socket, true);
  crs::bind(listen_socket, port_number);
  crs::listen(listen_socket);
  crs::out("host '%' waiting for connections on port '%'...\n",
           crs::gethostname(), port_number);
  for(;;)
  {
    const auto [dialog_socket, from_ip_addr, from_port]=
      crs::acceptfrom(listen_socket);
    crs::out("new connection from %:%\n",
             crs::formatIpv4Address(from_ip_addr), from_port);
    auto th=std::thread{dialog_thread, dialog_socket};
    th.detach();
  }
  crs::close(listen_socket);

  // ...

  return 0;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
