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
import glob, os, shutil, sys, time, string, warnings, datetime
from collections import OrderedDict
import numpy as np
if __name__ != '__main__':
    import lasagne

#----------------------------------------------------------------------------

def shape_to_str(shape):
    str = ['%d' % v if v else '?' for v in shape]
    return ', '.join(str) if len(str) else ''

#----------------------------------------------------------------------------

def generate_network_topology_info(layers):
    yield "%-14s%-20s%-10s%-20s%s" % ('LayerName', 'LayerType', 'Params', 'OutputShape', 'WeightShape')
    yield "%-14s%-20s%-10s%-20s%s" % (('---',) * 5)

    total_params = 0
    for layer in lasagne.layers.get_all_layers(layers):
        type_str = type(layer).__name__
        outshape = lasagne.layers.get_output_shape(layer)
        try:
            weights = layer.W.get_value()
        except:
            try:
                weights = layer.W_param.get_value()
            except:
                weights = np.zeros(())
        nparams = lasagne.layers.count_params(layer, trainable = True) - total_params

        weight_str = shape_to_str(weights.shape) if type_str != 'DropoutLayer' else 'p = %g' % layer.p
        yield "%-14s%-20s%-10d%-20s%s" % (layer.name, type_str, nparams, shape_to_str(outshape), weight_str)
        total_params += nparams

    yield "%-14s%-20s%-10s%-20s%s" % (('---',) * 5)
    yield "%-14s%-20s%-10d%-20s%s" % ('Total', '', total_params, '', '')

#----------------------------------------------------------------------------

def create_result_subdir(result_dir, run_desc):
    ordinal = 0
    for fname in glob.glob(os.path.join(result_dir, '*')):
        try:
            fbase = os.path.basename(fname)
            ford = int(fbase[:fbase.find('-')])
            ordinal = max(ordinal, ford + 1)
        except ValueError:
            pass

    result_subdir = os.path.join(result_dir, '%03d-%s' % (ordinal, run_desc))
    if os.path.isdir(result_subdir):
        return create_result_subdir(result_dir, run_desc) # Retry.
    if not os.path.isdir(result_subdir):
        os.makedirs(result_subdir)
    return result_subdir

#----------------------------------------------------------------------------

def export_sources(target_dir):
    os.makedirs(target_dir)
    for ext in ('py', 'pyproj', 'sln'):
        for fn in glob.glob('*.' + ext):
            shutil.copy2(fn, target_dir)
        if os.path.isdir('src'):
            for fn in glob.glob(os.path.join('src', '*.' + ext)):
                shutil.copy2(fn, target_dir)

#----------------------------------------------------------------------------

def export_run_details(fname):
    with open(fname, 'wt') as f:
        f.write('%-16s%s\n' % ('Host', config.host))
        f.write('%-16s%s\n' % ('User', config.user))
        f.write('%-16s%s\n' % ('Date', datetime.datetime.today()))
        f.write('%-16s%s\n' % ('CUDA device', config.cuda_device_number))
        f.write('%-16s%s\n' % ('Working dir', os.getcwd()))
        f.write('%-16s%s\n' % ('Executable', sys.argv[0]))
        f.write('%-16s%s\n' % ('Arguments', ' '.join(sys.argv[1:])))

#----------------------------------------------------------------------------

def export_config(fname):
    with open(fname, 'wt') as fout:
        for k, v in sorted(config.__dict__.iteritems()):
            if not k.startswith('_'):
                fout.write("%s = %s\n" % (k, str(v)))

#----------------------------------------------------------------------------

class GenericCSV(object):
    def __init__(self, fname, *fields):
        self.fields = fields
        self.fout = open(fname, 'wt')
        self.fout.write(string.join(fields, ',') + '\n')
        self.fout.flush()

    def add_data(self, *values):
        assert len(values) == len(self.fields)
        strings = [v if isinstance(v, str) else '%g' % v for v in values]
        self.fout.write(string.join(strings, ',') + '\n')
        self.fout.flush()

    def close(self):
        self.fout.close()

    def __enter__(self): # for 'with' statement
        return self

    def __exit__(self, *excinfo):
        self.close()

#----------------------------------------------------------------------------

def merge_csv_reports(result_dir):
    print 'Merging CSV reports in', result_dir
    print

    # List runs.

    subdirs = os.listdir(result_dir)
    max_digits = max([3] + [subdir.find('-') for subdir in subdirs if subdir[0] in '0123456789'])

    runs = []
    for subdir in subdirs:
        if subdir[0] in '0123456789':
            run_path = os.path.join(result_dir, subdir)
            if os.path.isdir(run_path):
                run_id = '0' * (max_digits - max(subdir.find('-'), 0)) + subdir
                runs.append((run_id, run_path))
    runs.sort()

    # Collect rows.

    all_rows = []
    for run_id, run_path in runs:
        print run_id
        run_rows = []
        for csv in glob.glob(os.path.join(run_path, '*.csv')):
            with open(csv, 'rt') as file:
                lines = [line.strip().split(',') for line in file.readlines()]
            run_rows += [OrderedDict([('RunID', run_id)] + zip(lines[0], line)) for line in lines[1:]]
            if len(lines) >= 2 and 'Epoch' in run_rows[-1] and run_rows[-1]['Epoch']:
                run_rows.append(OrderedDict(run_rows[-1]))
                run_rows[-1]['Epoch'] = ''
        all_rows += run_rows

    # Format output.

    fields = ('Stat', 'Value', 'RunID', 'Epoch')
    lines = []
    for row in all_rows:
        stats = [stat for stat in row.iterkeys() if stat not in fields]
        rest = [row.get(field, '') for field in fields[2:]]
        lines += [[stat, row[stat]] + rest for stat in stats]

    # Write CSV.

    fname = os.path.join(result_dir, 'merged.csv')
    print
    print "Writing", fname

    with open(fname, 'wt') as file:
        file.write(string.join(fields, ',') + '\n')
        for line in lines:
            file.write(string.join(line, ',') + '\n')

    print 'Done.'
    print

#----------------------------------------------------------------------------

if __name__ == '__main__':
    print
    if len(sys.argv) != 2 or sys.argv[1].startswith('-'):
        print "Usage: python %s <result_dir>" % sys.argv[0]
    else:
        merge_csv_reports(sys.argv[1])

#----------------------------------------------------------------------------
