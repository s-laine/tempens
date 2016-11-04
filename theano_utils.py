#  Copyright (c) 2016, NVIDIA Corporation
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of NVIDIA Corporation nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import config
import os, sys, warnings
import theano

#----------------------------------------------------------------------------

# Check for common problems in a compiled Theano function.
def analyze_function(func, verbose = False):
    assert isinstance(func, theano.compile.Function)
    topo = func.maker.fgraph.toposort()

    # Print stats.

    if verbose:
        op_names = [type(apply.op).__name__ for apply in topo]
        op_dict = {op: 0 for op in op_names}
        for op in op_names:
            op_dict[op] += 1

        op_list = op_dict.items()
        op_list.sort(key = lambda x: -x[1])

        print
        for op, num in op_list:
            print "  %-8d%s" % (num, op)
        print

    # Check for float64 use.

    for apply in topo:
        dtype = getattr(apply.outputs[0].type, 'dtype', '')
        acc_dtype = getattr(apply.op, 'acc_dtype', '')
        if dtype == 'float64' or acc_dtype == 'float64':
            print 'WARNING: Theano float64:', apply
            if verbose:
                print
                theano.printing.debugprint(apply)
                print

    # Check for excess GPU=>CPU transfers.

    for apply in topo:
        op = type(apply.op).__name__
        if op == 'HostFromGpu':
            for parent in topo:
                parent_inputs = [var.owner for var in parent.inputs]
                if apply in parent_inputs:
                    print 'WARNING: Theano CPU fallback:', parent
                    if verbose:
                        print
                        theano.printing.debugprint(parent)
                        print

#----------------------------------------------------------------------------

# Compile and check Theano function.
def function(*args, **kwargs):
    func = theano.function(*args, **kwargs)
    analyze_function(func, verbose = False)
    return func

#----------------------------------------------------------------------------
