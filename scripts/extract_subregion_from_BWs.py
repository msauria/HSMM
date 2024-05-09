#!/usr/bin/env python

import sys

import numpy
import pyBigWig


def main():
    region, out_fname = sys.argv[1:3]
    in_fnames = [x.split(',') for x in sys.argv[3:]]
    chrom = region.split(':')[0]
    start = int(region.split(':')[-1].split('-')[0])
    end = int(region.split(':')[-1].split('-')[1])
    dtype = [('chr', '<U5'), ('start', numpy.int32), ('end', numpy.int32)]
    for fname, name in in_fnames:
        dtype.append((name, numpy.int32))
    data = numpy.zeros((end - start) // 100, dtype=numpy.dtype(dtype))
    data['chr'][:] = chrom
    data['start'][:] = numpy.arange(data.shape[0]) * 100 + start
    data['end'][:] = data['start'] + 100
    for fname, name in in_fnames:
        bw = pyBigWig.open(fname, 'r')
        entries = numpy.array(list(bw.intervals(chrom, start, end)), dtype=numpy.dtype([
            ('start', numpy.int32), ('end', numpy.int32), ('value', numpy.float64)]))
        sizes = (entries['end'] - entries['start'])
        where = numpy.where(sizes == 100)[0]
        indices = numpy.searchsorted(data['start'], entries['start'][where], side='left')
        data[name][indices] = entries['value'][where].astype(numpy.int32)
        where = numpy.where(sizes > 100)[0]
        indices = numpy.searchsorted(data['start'], entries['start'][where], side='left')
        indices2 = numpy.searchsorted(data['start'], entries['end'][where], side='left')
        for i, x in enumerate(where):
            data[name][indices[i]:indices2[i]] = entries['value'][x].astype(numpy.int32)
        bw.close()
    numpy.save(out_fname, data)

main()