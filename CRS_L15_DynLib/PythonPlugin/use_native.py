#!/usr/bin/env python
# -*- coding: utf-8 -*-
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import sys
import os
import ctypes

def out(fmt, *args, **kw):
  sys.stdout.write(fmt.format(*args, **kw))

def main():
  out('~~ loading the native library ~~\n')
  lib_path=os.path.join(os.path.curdir, # search in current directory
                        system_library_name('Native'))
  native_lib=ctypes.CDLL(os.path.abspath(lib_path))
  out('--> {}\n', native_lib)
  step_01(native_lib)
  step_02(native_lib)
  step_03(native_lib)
  step_04(native_lib)
  step_05(native_lib)

def system_library_name(lib_name):
  if sys.platform=='win32':
    return lib_name+'.dll'
  elif sys.platform=='darwin':
    return 'lib'+lib_name+'.dylib'
  else:
    return 'lib'+lib_name+'.so'

def step_01(native_lib):
  out('\n~~ step 01 ~~ no argument/result\n')
  #~~ look for the function into the library ~~
  # ... À COMPLÉTER ...
  #~~ use this function ~~
  # ... À COMPLÉTER ...

def step_02(native_lib):
  out('\n~~ step 02 ~~ simple types as arguments/result\n')
  #~~ look for the function into the library ~~
  # ... À COMPLÉTER ...
  #~~ use this function ~~
  # ... À COMPLÉTER ...

def step_03(native_lib):
  out('\n~~ step 03 ~~ access by reference\n')
  #~~ look for the function into the library ~~
  # ... À COMPLÉTER ...
  #~~ use this function ~~
  # ... À COMPLÉTER ...

def show_array(array):
  return '['+ (', '.join((str(e) for e in array)))+']'

def step_04(native_lib):
  out('\n~~ step 04 ~~ access array\n')
  #~~ look for the function into the library ~~
  # ... À COMPLÉTER ...
  #~~ use this function ~~
  # ... À COMPLÉTER ...

def step_05(native_lib):
  out('\n~~ step 05 ~~ create/use/destroy something\n')
  #~~ look for the functions into the library ~~
  create_Something=native_lib['create_Something']
  create_Something.restype=ctypes.c_void_p
  create_Something.argtypes=[ctypes.c_int] # int count
  use_Something=native_lib['use_Something']
  use_Something.argtypes=[ctypes.c_void_p, # const Something &something
                          ctypes.c_void_p, # const char * &out_c_name
                          ctypes.c_void_p, # const double * &out_values
                          ctypes.c_void_p] # int &out_value_count
  destroy_Something=native_lib['destroy_Something']
  destroy_Something.argtypes=[ctypes.c_void_p] # Something &something
  #~~ use these functions ~~
  something=create_Something(5)
  out_c_name=ctypes.c_char_p()
  out_values=ctypes.c_void_p()
  out_value_count=ctypes.c_int()
  use_Something(something,
                ctypes.byref(out_c_name),
                ctypes.byref(out_values),
                ctypes.byref(out_value_count))
  values=(ctypes.c_double*out_value_count.value).from_address(out_values.value)
  out('--> {} {}\n', out_c_name.value, show_array(values))
  destroy_Something(something)

if __name__=='__main__':
  main()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
