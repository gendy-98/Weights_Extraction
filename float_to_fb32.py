# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 22:45:04 2021

@author: ahmed
"""
import struct
import numpy as np

def binary(num):
    return ''.join(bin(int(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', num))


def binary_discribtion(num):
    # Struct can provide us with the float packed into bytes. The '!' ensures that
    # it's in network byte order (big-endian) and the 'f' says that it should be
    # packed as a float. Alternatively, for double-precision, you could use 'd'.
    packed = struct.pack('!f', num)
    print ('Packed: %s' % repr(packed))

    # For each character in the returned string, we'll turn it into its corresponding
    # integer code point
    # 
    # [62, 163, 215, 10] = [ord(c) for c in '>\xa3\xd7\n']
    integers = [int(c) for c in packed]
    print ('Integers: %s' % integers)

    # For each integer, we'll convert it to its binary representation.
    binaries = [bin(i) for i in integers]
    print ('Binaries: %s' % binaries)

    # Now strip off the '0b' from each of these
    stripped_binaries = [s.replace('0b', '') for s in binaries]
    print ('Stripped: %s' % stripped_binaries)

    # Pad each byte's binary representation's with 0's to make sure it has all 8 bits:
    #
    # ['00111110', '10100011', '11010111', '00001010']
    padded = [s.rjust(8, '0') for s in stripped_binaries]
    print ('Padded: %s' % padded)

    # At this point, we have each of the bytes for the network byte ordered float
    # in an array as binary strings. Now we just concatenate them to get the total
    # representation of the float:
    return ''.join(padded)

def float2fix_complement(val, width, precision):
    integer = abs(int(val * 2 ** precision))
    if integer == 0:
        fix_str = bin(0).replace('0b', '').rjust(width,'0')
    else:
        if val >= 0:
            fix_str = bin(integer).replace('0b', '').rjust(width, '0')
        else:
            integer = 2 ** (width-1) - integer
            fix_str = '1'+bin(integer).replace('0b', '').rjust(width-1, '0')
    return fix_str

def padding(pad_xoffset, pad_yoffset, refrerance_arr, input_array):
    result = np.zeros_like(refrerance_arr)
    
    
    
    x_offset = pad_xoffset  # 0 would be what you wanted
    y_offset = pad_yoffset  # 0 in your case
    result[x_offset:input_array.shape[0]+x_offset,y_offset:input_array.shape[1]+y_offset] = input_array
    return result







def float2fix_signbit(val, width, precision):
    integer = abs(int(val * 2 ** precision))
    if val < 0:
        val = abs(val)
        integer = abs(int(val * 2 ** precision))
    if val >= 0:
        fix_str = bin(integer).replace('0b', '').rjust(width, '0')
    else:
        fix_str = '1'+bin(integer).replace('0b', '').rjust(width-1, '0')
    return fix_str	