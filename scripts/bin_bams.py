#!/usr/bin/env python

import sys

import numpy
import pysam
# import pyBigWig

binsize = 3000

def main():
    bam_fname, out_fname = sys.argv[1:3]
    bam = pysam.AlignmentFile(bam_fname, 'r')
    header = bam.header['SQ']
    chroms = [(x['SN'], x['LN']) for x in header if x['SN'].lstrip('chr').isdigit() or x['SN'] == 'chrX']
    pe = 0
    for i, read in enumerate(bam):
        if i == 1000:
            break
        pe += read.is_paired
    bam.close()
    if pe > 500:
        data = process_pe(bam_fname, out_fname)
    else:
        data = process_se(bam_fname, out_fname)
    write_bigwig(data, chroms, out_fname)

def process_pe(bam_fname, out_fname):
    data = {}
    unmated = {}
    bam = pysam.AlignmentFile(bam_fname, 'r')
    for read in bam:
        if not read.is_paired or read.is_unmapped or read.mate_is_unmapped:
            continue
        if read.is_supplementary:
            continue
        if read.mapping_quality < 20:
            continue
        ref_seq = read.reference_name
        if not ref_seq.lstrip('chr').isdigit() and ref_seq != 'chrX':
            continue
        query_seq = read.query_name
        data.setdefault(ref_seq, [])
        if read.is_forward:
            coord = read.reference_start
        else:
            coord = -read.reference_end + 1
        if query_seq in unmated:
            if coord > 0:
                data[ref_seq].append((coord, unmated[ref_seq][query_seq]))
            else:
                data[ref_seq].append((unmated[ref_seq][query_seq], coord))
            del unmated[ref_seq][query_seq]
        else:
            unmated.setdefault(ref_seq, {})
            unmated[ref_seq][query_seq] = coord
    bam.close()
    del unmated
    for chrom in data:
        data[chrom] = numpy.array(data[chrom], numpy.int32)
        data[chrom] = data[chrom][numpy.where(numpy.logical_and(
            data[chrom][:, 0] > 0, data[chrom][:, 1] < 0))[0], :]
        sizes = -data[chrom][:, 1] - data[chrom][:, 0]
        data[chrom] = data[chrom][numpy.where(numpy.logical_and(
            sizes >= 50, sizes <= 1000))[0], :]
        data[chrom] = (data[chrom][:, 0] + data[chrom][:, 1]) // 2
    return data

def process_se(bam_fname, out_fname):
    data = {}
    bam = pysam.AlignmentFile(bam_fname, 'r')
    for i, read in enumerate(bam):
        if read.is_paired or read.is_unmapped:
            continue
        if read.is_supplementary:
            continue
        if read.mapping_quality < 20:
            continue
        ref_seq = read.reference_name
        if not ref_seq.lstrip('chr').isdigit() and ref_seq != 'chrX':
            continue
        query_seq = read.query_name
        data.setdefault(ref_seq, [])
        if read.is_forward:
            data[ref_seq].append(read.reference_start)
        else:
            data[ref_seq].append(-read.reference_end + 1)
    peaks = []
    peak_indices = [0]
    chroms = list(data.keys())
    chroms.sort()
    valid = {}
    for chrom in chroms:
        data[chrom] = numpy.array(data[chrom], numpy.int32)
        data[chrom] = data[chrom][numpy.argsort(numpy.abs(data[chrom]))]
        valid[chrom] = numpy.ones(data[chrom].shape[0], bool)
        binned = numpy.bincount(numpy.abs(data[chrom]) // binsize)
        peaks.append(binned)
        peak_indices.append(peak_indices[-1] + len(binned))
    peaks = numpy.concatenate(peaks, axis=0)
    peak_indices = numpy.array(peak_indices)
    order = numpy.argsort(peaks)[::-1]
    count = 0
    pos = 0
    corrs = numpy.zeros(1000, numpy.float32)
    while count < 1000:
        chrint = numpy.searchsorted(peak_indices, order[pos], side='right') - 1
        chrom = chroms[chrint]
        mid = (order[pos] - peak_indices[chrint]) * binsize + binsize // 2
        s, e =  numpy.searchsorted(numpy.abs(data[chrom]), numpy.r_[mid - binsize, mid + binsize])
        if numpy.sum(valid[chrom][s:e] == 0) > 0:
            pos += 1
            continue
        binned = numpy.bincount(numpy.abs(data[chrom][s:e]) - (mid - binsize), minlength=binsize*2)
        windowed = bin_window(binned, binsize+1)
        best = mid - binsize // 2 + numpy.argmax(windowed)
        start = best - (binsize // 2 + 1000)
        end = best + (binsize // 2 + 1000)
        s, e =  numpy.searchsorted(numpy.abs(data[chrom]), numpy.r_[start, end])
        if numpy.sum(valid[chrom][s:e] == False) > 0:
            pos += 1
            continue
        pwhere = numpy.where(data[chrom][s:e] > 0)[0]
        pbinned = numpy.bincount(data[chrom][s:e][pwhere] - start, minlength=(end-start))
        pwindow = bin_window(pbinned[:-500], binsize + 1)
        nwhere = numpy.where(data[chrom][s:e] < 0)[0]
        nbinned = numpy.bincount(-data[chrom][s:e][nwhere] - start, minlength=(end-start))
        nwindow = bin_window(nbinned[500:], binsize + 1)
        valid[chrom][s:e] = False
        for i in range(10, 1000):
            forward = pwindow[500 - i // 2:1500 - i // 2]
            reverse = nwindow[(i - i // 2):(i - i // 2) + 1000]
            tmp = numpy.corrcoef(forward, reverse)[0, 1]
            if not numpy.isnan(tmp):
                corrs[i] += tmp
        count += 1
        pos += 1
    corrs /= 1000
    best = numpy.argmax(corrs)
    print(best, corrs[best])
    mids = {}
    for chrom in chroms:
        h1 = best // 2
        h2 = best - h1
        mids[chrom] = numpy.zeros(data[chrom].shape[0], numpy.int32)
        pwhere = numpy.where(data[chrom] > 0)[0]
        mids[chrom][pwhere] = data[chrom][pwhere] + h1
        nwhere = numpy.where(data[chrom] < 0)[0]
        mids[chrom][nwhere] = -data[chrom][nwhere] - h2
    return mids

def bin_window(data, window):
    bdata = numpy.zeros(data.shape[0] - window + 1, data.dtype)
    bdata[0] = numpy.sum(data[:window])
    for i in range(1, bdata.shape[0]):
        bdata[i] = bdata[i - 1] - data[i - 1] + data[i + window - 1]
    return bdata

def write_bigwig(data, chroms, fname):
    output = open(fname, 'w')
    for chrom, maxlen in chroms:
        counts = numpy.bincount(data[chrom] // 100)
        breaks = numpy.r_[0, numpy.where(numpy.diff(counts > 0))[0] + 1, counts.shape[0]]
        header = True
        for i in range(breaks.shape[0] - 1):
            s, e = breaks[i:i+2]
            if counts[s] == 0:
                N = (e - s) * 100
                if N > 6000:
                    output.write(f"fixedStep chrom={chrom} start={s * 100 + 1} step={N} span={N}\n0\n")
                    header = True
                else:
                    output.write("0\n" * (e - s))
            else:
                if header:
                    output.write(f"fixedStep chrom={chrom} start={s * 100 + 1} step=100 span=100\n")
                    header = False
                for j in range(s, e):
                    output.write(f"{counts[j]}\n")
    output.close()
    return



main()































