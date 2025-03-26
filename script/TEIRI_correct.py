import time
import os
import numpy as np
import pandas as pd
import tabix
import torch
import selene_sdk
import pyBigWig
import argparse
from torch import nn
from scipy.special import softmax
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
from torch.optim.lr_scheduler import ReduceLROnPlateau
from selene_sdk.sequences import Genome
from numpy import log10


class ConvBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio=2, fused=True):
        super(ConvBlock, self).__init__()
        hidden_dim = round(inp * expand_ratio)
        self.conv = nn.Sequential(
            nn.Conv1d(inp, hidden_dim, 9, 1, padding=4, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=False),
            nn.Conv1d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm1d(oup),
        )


    def forward(self, x):
        return x + self.conv(x)



class PuffinD(nn.Module):
    def __init__(self):
        """
        Parameters
        ----------
        """
        super(PuffinD, self).__init__()
        self.uplblocks = nn.ModuleList([
                nn.Sequential(
                nn.Conv1d(6, 64, kernel_size=17, padding=8),
            nn.BatchNorm1d(64)),

            nn.Sequential(
            nn.Conv1d(64, 96, stride=4, kernel_size=17, padding=8),
            nn.BatchNorm1d(96)),

            nn.Sequential(
                nn.Conv1d(96, 128, stride=4, kernel_size=17, padding=8),
            nn.BatchNorm1d(128)),

            nn.Sequential(
            nn.Conv1d(128, 128, stride=5, kernel_size=17, padding=8),
            nn.BatchNorm1d(128)),

            nn.Sequential(
            nn.Conv1d(128, 128, stride=5, kernel_size=17, padding=8),
            nn.BatchNorm1d(128)),

            nn.Sequential(
            nn.Conv1d(128, 128, stride=5, kernel_size=17, padding=8),
            nn.BatchNorm1d(128)),

            nn.Sequential(
            nn.Conv1d(128, 128, stride=2, kernel_size=17, padding=8),
            nn.BatchNorm1d(128)),

        ])

        self.upblocks = nn.ModuleList([
            nn.Sequential(
            ConvBlock(64, 64, fused=True),
            ConvBlock(64, 64, fused=True)),

            nn.Sequential(
            ConvBlock(96, 96, fused=True),
            ConvBlock(96, 96, fused=True)),

            nn.Sequential(
                ConvBlock(128, 128, fused=True),
            ConvBlock(128, 128, fused=True)),

            nn.Sequential(
            ConvBlock(128, 128, fused=True),
            ConvBlock(128, 128, fused=True)),

            nn.Sequential(
            ConvBlock(128, 128, fused=True),
            ConvBlock(128, 128, fused=True)),

            nn.Sequential(
            ConvBlock(128, 128, fused=True),
            ConvBlock(128, 128, fused=True)),

            nn.Sequential(
            ConvBlock(128, 128, fused=True),
            ConvBlock(128, 128, fused=True)),

        ])

        self.downlblocks = nn.ModuleList([
    
            nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 128, kernel_size=17, padding=8),
            nn.BatchNorm1d(128)),

            nn.Sequential(
            nn.Upsample(scale_factor=5),
            nn.Conv1d(128, 128, kernel_size=17, padding=8),
            nn.BatchNorm1d(128)),

            nn.Sequential(
            nn.Upsample(scale_factor=5),
            nn.Conv1d(128, 128, kernel_size=17, padding=8),
            nn.BatchNorm1d(128)),

            nn.Sequential(
            nn.Upsample(scale_factor=5),
            nn.Conv1d(128, 128, kernel_size=17, padding=8),
            nn.BatchNorm1d(128)),

            nn.Sequential(
                nn.Upsample(scale_factor=4),
            nn.Conv1d(128, 96, kernel_size=17, padding=8),
            nn.BatchNorm1d(96)),

            nn.Sequential(
            nn.Upsample(scale_factor=4),
            nn.Conv1d(96, 64, kernel_size=17, padding=8),
            nn.BatchNorm1d(64)),


        ])

        self.downblocks = nn.ModuleList([
                nn.Sequential(
            ConvBlock(128, 128, fused=True),
            ConvBlock(128, 128, fused=True)),

            nn.Sequential(
            ConvBlock(128, 128, fused=True),
            ConvBlock(128, 128, fused=True)),

            nn.Sequential(
            ConvBlock(128, 128, fused=True),
            ConvBlock(128, 128, fused=True)),

            nn.Sequential(
            ConvBlock(128, 128, fused=True),
            ConvBlock(128, 128, fused=True)),

            nn.Sequential(
                ConvBlock(96, 96, fused=True),
            ConvBlock(96, 96, fused=True)),

            nn.Sequential(
            ConvBlock(64, 64, fused=True),
            ConvBlock(64, 64, fused=True))


        ])

        self.uplblocks2 = nn.ModuleList([
    
            nn.Sequential(
            nn.Conv1d(64, 96, stride=4, kernel_size=17, padding=8),
            nn.BatchNorm1d(96)),

            nn.Sequential(
                nn.Conv1d(96, 128, stride=4, kernel_size=17, padding=8),
            nn.BatchNorm1d(128)),

            nn.Sequential(
            nn.Conv1d(128, 128, stride=5, kernel_size=17, padding=8),
            nn.BatchNorm1d(128)),

            nn.Sequential(
            nn.Conv1d(128, 128, stride=5, kernel_size=17, padding=8),
            nn.BatchNorm1d(128)),

            nn.Sequential(
            nn.Conv1d(128, 128, stride=5, kernel_size=17, padding=8),
            nn.BatchNorm1d(128)),

            nn.Sequential(
            nn.Conv1d(128, 128, stride=2, kernel_size=17, padding=8),
            nn.BatchNorm1d(128)),

        ])

        self.upblocks2 = nn.ModuleList([
    
            nn.Sequential(
            ConvBlock(96, 96, fused=True),
            ConvBlock(96, 96, fused=True)),

            nn.Sequential(
                ConvBlock(128, 128, fused=True),
            ConvBlock(128, 128, fused=True)),

            nn.Sequential(
            ConvBlock(128, 128, fused=True),
            ConvBlock(128, 128, fused=True)),

            nn.Sequential(
            ConvBlock(128, 128, fused=True),
            ConvBlock(128, 128, fused=True)),

            nn.Sequential(
            ConvBlock(128, 128, fused=True),
            ConvBlock(128, 128, fused=True)),

            nn.Sequential(
            ConvBlock(128, 128, fused=True),
            ConvBlock(128, 128, fused=True)),

        ])

        self.downlblocks2 = nn.ModuleList([
    
            nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 128, kernel_size=17, padding=8),
            nn.BatchNorm1d(128)),

            nn.Sequential(
            nn.Upsample(scale_factor=5),
            nn.Conv1d(128, 128, kernel_size=17, padding=8),
            nn.BatchNorm1d(128)),

            nn.Sequential(
            nn.Upsample(scale_factor=5),
            nn.Conv1d(128, 128, kernel_size=17, padding=8),
            nn.BatchNorm1d(128)),

            nn.Sequential(
            nn.Upsample(scale_factor=5),
            nn.Conv1d(128, 128, kernel_size=17, padding=8),
            nn.BatchNorm1d(128)),

            nn.Sequential(
                nn.Upsample(scale_factor=4),
            nn.Conv1d(128, 96, kernel_size=17, padding=8),
            nn.BatchNorm1d(96)),

            nn.Sequential(
            nn.Upsample(scale_factor=4),
            nn.Conv1d(96, 64, kernel_size=17, padding=8),
            nn.BatchNorm1d(64)),


        ])

        self.downblocks2 = nn.ModuleList([
                nn.Sequential(
            ConvBlock(128, 128, fused=True),
            ConvBlock(128, 128, fused=True)),

            nn.Sequential(
            ConvBlock(128, 128, fused=True),
            ConvBlock(128, 128, fused=True)),

            nn.Sequential(
            ConvBlock(128, 128, fused=True),
            ConvBlock(128, 128, fused=True)),

            nn.Sequential(
            ConvBlock(128, 128, fused=True),
            ConvBlock(128, 128, fused=True)),

            nn.Sequential(
                ConvBlock(96, 96, fused=True),
            ConvBlock(96, 96, fused=True)),

            nn.Sequential(
            ConvBlock(64, 64, fused=True),
            ConvBlock(64, 64, fused=True))


        ])
        self.final =  nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 1, kernel_size=1),
            nn.Softplus())

    def forward(self, x):
        """Forward propagation of a batch.
        """
        out = x
        encodings = []
        for i, lconv, conv in zip(np.arange(len(self.uplblocks)), self.uplblocks, self.upblocks):
            lout = lconv(out)
            out = conv(lout)
            encodings.append(out)

        encodings2 = [out]
        for enc, lconv, conv in zip(reversed(encodings[:-1]), self.downlblocks, self.downblocks):
            lout = lconv(out)
            out = conv(lout)
            out = enc + out
            encodings2.append(out)

        encodings3 = [out]
        for enc, lconv, conv in zip(reversed(encodings2[:-1]), self.uplblocks2, self.upblocks2):
            lout = lconv(out)
            out = conv(lout)
            out = enc + out
            encodings3.append(out)

        for enc, lconv, conv in zip(reversed(encodings3[:-1]), self.downlblocks2, self.downblocks2):
            lout = lconv(out)
            out = conv(lout)
            out = enc + out

        out = self.final(out)
        return out



def process_gtf(gtf_path):
    transcript = {}
    print((gtf_path+"_load"))
    with open(gtf_path) as file:
        for eachline in file:
            if not eachline.strip().startswith('#'):
                temp = eachline.strip().split("\t")
                type = temp[2]
                if type == "exon":
                    anno = temp[8].split("\"; ")
                    for transcriptinfo in anno:
                        if "transcript_id" in transcriptinfo:
                            transcriptid = transcriptinfo.split("\"")[1]
                            break
                    id = "\t".join([temp[0], str(temp[6]), transcriptid])
                    if id not in transcript:
                        transcript[id] = []
                    transcript[id].append(int(temp[3]))
                    transcript[id].append(int(temp[4]))
    tss_gtf = {} 
    for transciptID in transcript:
        id = transciptID.split("\t")
        exons=transcript[transciptID]
        exons.sort() 
        num_exons=int(len(exons)/2)
        chr_s = "\t".join([id[0],id[1]])
        
        if num_exons>=2 :
            if chr_s not in tss_gtf:
                tss_gtf[chr_s]=[]
            if id[1]=="+":
                exon1_end=exons[1]
            if id[1]=="-":
                exon1_end=exons[-2]
            tss_gtf[chr_s].append(exon1_end)
    
    for chr_s in tss_gtf:
        tss_gtf[chr_s] = list(set(tss_gtf[chr_s]))

    return tss_gtf


def process_bed12(bed12_path):
    tss_count = {}
    lib_size = 0

    with open(bed12_path) as nanopore:
        for eachline in nanopore:
            if not eachline.strip().startswith('#'):
                temp = eachline.strip().split("\t")
                chr_s = "\t".join([temp[0],str(temp[5])])
                exon_length = temp[10].split(",")
                exon_start = temp[11].split(",")
                if int(temp[9]) >=2 :
                    if chr_s not in tss_count:
                        tss_count[chr_s]={}
                    if temp[5] == "+":
                        tss = str(int(temp[1]) + int(exon_start[0]) + 1)
                        exon1_end = str(int(temp[1]) + int(exon_start[0]) + int(exon_length[0]))
                    if temp[5] == "-":
                        tss = str(int(temp[1]) + int(exon_start[int(temp[9])-1]) + int(exon_length[int(temp[9])-1]))
                        exon1_end = str(int(temp[1]) + int(exon_start[int(temp[9])-1]) + 1)

                    if exon1_end not in tss_count[chr_s]:
                        tss_count[chr_s][exon1_end] = {}
                    if tss not in tss_count[chr_s][exon1_end]:
                        tss_count[chr_s][exon1_end][tss] = 0
                    tss_count[chr_s][exon1_end][tss] += 1
                    lib_size += 1

    lib_size = lib_size/1e6

    for chr_s in tss_count:
        for exon1_end in tss_count[chr_s]:
            for tss in tss_count[chr_s][exon1_end]:
                tss_count[chr_s][exon1_end][tss] = tss_count[chr_s][exon1_end][tss]/lib_size

    return tss_count


def data_load(batch,window_length,genome,bw_load):
    sequences = []
    labels = []
    mask = []
    
    for j in range(len(batch)):
        chr,strand,position,sample,label = batch.iloc[j]
        if strand =="+":
            start = int(position) - window_length - 1
            end = int(position) + window_length - 1
            label_zero = np.zeros((1, (window_length*2 ) ))
            if start > 0 and end < chr_len[chr] :
                seq = genome.get_encoding_from_coords(chr, start, end)
                for bw_type in bw_types:
                    temp = log10(np.array(bw_load[sample][bw_type].values(chr, start, end))+1)
                    seq = np.column_stack((seq, temp))
                #for bw_type in ['plus','minus']:
                #    temp = np.array(tss_bw[bw_type].values(chr, start, end))
                #    seq = np.column_stack((seq, temp))
                sequences.append(np.transpose(seq))
                if label !=0 :
                    for tss in label:
                        location =  int(tss) - int(position) + window_length
                        label_zero[0, location] = float(log10(float(label[tss])+1))
                labels.append(label_zero)

        elif strand == "-":
            start = int(position) - window_length 
            end = int(position) + window_length 
            label_zero = np.zeros((1, (window_length*2 ) ))
            if start > 0 and end < chr_len[chr] :
                seq = genome.get_encoding_from_coords(chr, start, end)[::-1, ::-1]
                for bw_type in bw_types:
                    temp = log10(np.array(bw_load[sample][bw_type].values(chr, start, end))+1)[::-1]
                    seq = np.column_stack((seq, temp))
                sequences.append(np.transpose(seq))
                if label !=0 :
                    for tss in label:
                        location =  int(tss) - int(position) + window_length - 1
                        label_zero[0, location] = float(log10(float(label[tss])+1))
                labels.append(label_zero[:,::-1])
        
        mask.append(np.concatenate([np.ones(window_length), np.zeros(1),  np.zeros(window_length-1)]).reshape(1,window_length *2))

    model_input = np.nan_to_num(np.stack(sequences),nan=0)
    model_labels = np.stack(labels)
    masks = np.stack(mask)

    return model_input,model_labels,masks



def max_score_tss(arr, window_length):
    closest_indices = []
    for i in range(arr.shape[0]):
        max_value = np.max(arr[i])
        max_indices = np.where(arr[i] == max_value)[1]
        distances = np.abs(max_indices - window_length)
        closest_index = max_indices[np.argmin(distances)]
        closest_indices.append(closest_index)
    closest_indices_array = np.array(closest_indices).reshape(-1, 1)
    return closest_indices_array

def PseudoPoissonKL(lpred, ltarget):
    ltarget = torch.clamp(ltarget, min=0) 
    return (ltarget * torch.log((ltarget+1e-10)/(lpred+1e-10)) + lpred - ltarget)




def tss_correct(gtf_path,predict):

    transcripts = {}
    print((gtf_path+"_load"))
    with open(gtf_path) as file:
        for eachline in file:
            if not eachline.strip().startswith('#'):
                temp = eachline.strip().split("\t")
                type = temp[2]
                if type == "exon":
                    anno = temp[8].split("\"; ")
                    for transcriptinfo in anno:
                        if "transcript_id" in transcriptinfo:
                            transcriptid = transcriptinfo.split("\"")[1]
                            break
                    id = "\t".join([temp[0], str(temp[6]), transcriptid])
                    if id not in transcripts:
                        transcripts[id] = []
                    transcripts[id].append(int(temp[3]))
                    transcripts[id].append(int(temp[4]))

    correct_transcripts = {}


    for ID in transcripts:
        exons = transcripts[ID]
        id = ID.split("\t")
        chrom = id[0]
        strand = id[1]
        transcriptid = id[2] 
        chr_s = "\t".join([id[0],id[1]])
        num_exons=int(len(exons)/2)
        exons.sort()
        if chr_s in predict:
            if strand=="+" or strand=="-":
                if num_exons>=2 :
                    if strand=="+":
                        if exons[1] in predict[chr_s]:
                            updated_tss = predict[chr_s][exons[1]]
                            if updated_tss != 0 :
                                exons[0] = updated_tss
                                correct_transcripts[ID] = exons
                    elif strand=="-":
                        if exons[-2] in predict[chr_s]:
                            updated_tss = predict[chr_s][exons[-2]]
                            if updated_tss != 0 :
                                exons[-1] = updated_tss
                                correct_transcripts[ID] = exons   
    
    return correct_transcripts

def write_exon(exons,transcriptid,chr_,strand):
    exons = list(map(int, exons))
    num_exons=int(len(exons)/2)
    exons.sort()
    exons = list(map(str, exons))
    stringreturn=[chr_, "TE_RNA","transcript" , exons[0] ,exons[-1],".", strand,".",str("transcript_id \""+ str(transcriptid) + "\";") ]
    stringreturn = "\t".join([str(x) for x in stringreturn])
    yield stringreturn
    for i in range(0,num_exons):
        stringreturn=[chr_, "TE_RNA","exon" , exons[2*i] ,exons[2*i+1],".", strand,".",str("transcript_id \""+ str(transcriptid) + "\";") ]
        stringreturn = "\t".join([str(x) for x in stringreturn])
        yield stringreturn

def write_gtf(transcripts,output_file):
    transcriptid = 0

    with open(output_file, "w") as f_out:
        for ID in transcripts:
            exons = transcripts[ID]
            id = ID.split("\t")
            chrom = id[0]
            strand = id[1]
            transcriptid = id[2] 
            for stringreturn in write_exon(exons,transcriptid,chrom,strand):
                f_out.write(stringreturn + "\n")

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='TEIRI_correct')
    parser.add_argument('--input', help="Prefix of input files (required)")
    parser.add_argument('--short_path', default=".", type=str, help="Path to input files (default: '.')")
    parser.add_argument('--chr_len', help="Chromosome length file (required)")
    parser.add_argument('--window_length', default=10000, type=int, help="Window size for TSS prediction (default:10000)")
    parser.add_argument('-g', '--ref_fa', help="Reference fasta file (required)")
    parser.add_argument('-m', '--model_path', help="Path to the deep-learning model (required)")
    parser.add_argument( '-p','--prefix', default="TEIRI",type=str, help="Prefix for ouput files (default: TEIRI)")
    args = parser.parse_args()

    batch_size = 64
    sl_match = {}
    long_read_sample = []
    long_tss_count = {}
    short_tss = {}
    bw_load = {}
    predict_outcome = {}


    genome = Genome(input_path=args.ref_fa)

    chr_len = {}

    with open(args.chr_len) as file:
        for eachline in file:
            temp = eachline.strip().split("\t")
            chr_len[temp[0]] = int(temp[1])


    file_path = args.short_path + "/" + args.input + "_stringtie.gtf"
    short_tss[args.input] = process_gtf(file_path)


    bw_types = ['_CPM_unique','_CPM_mult'] #,'_plus_acceptor','_plus_donor','_minus_acceptor','_minus_donor']
    for sample in short_tss:
        bw_load [sample] ={}
        for bw_type in bw_types:
            file_path = args.short_path + "/" + sample + bw_type + ".bw"
            bw_load [sample][bw_type] = pyBigWig.open(file_path)


    df = {'chr': [], 'strand': [], 'position': [], 'sample': [], 'label': []}

    for sample in short_tss:
        for chr_s in short_tss[sample]:
            for exon1_end in short_tss[sample][chr_s]:
                chr , strand = chr_s.split("\t")
                label = 0
                df['chr'].append(chr)         
                df['strand'].append(strand) 
                df['position'].append(exon1_end) 
                df['sample'].append(sample) 
                df['label'].append(label) 

    df = pd.DataFrame(df)
    print(df.shape[0])


    model = torch.load(args.model_path)
    model.eval()
    


    filtered_df = df[df['label']==0 ]
    filtered_df = filtered_df[filtered_df['chr'].isin(['chr1', 'chr2', 'chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13' \
        ,'chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chr22','chrX','chrY'])]

    print(filtered_df.shape[0])

    output_file_path = args.prefix + "_predict.tsv"
    with open(output_file_path, "w")  as file:
        with torch.no_grad():            

            for k in range(0, len(filtered_df), batch_size):
                valid_batch = filtered_df.iloc[k:k+batch_size]
                valid_input,valid_labels,valid_mask =data_load(valid_batch,args.window_length,genome,bw_load)
                pred = model(torch.FloatTensor(valid_input).cuda()).cpu().detach().numpy()
                pred_max_indices = max_score_tss(pred*valid_mask, args.window_length) 
                
                for i in range( valid_input.shape[0]):
                    chr,strand,position,sample,label = valid_batch.iloc[i]
                    chr_s = "\t".join([chr,strand])
                    if chr_s not in predict_outcome:
                        predict_outcome[chr_s] = {}
                    if strand == "+":
                        predict_outcome[chr_s][position] = position + pred_max_indices[i,0]  - args.window_length
                    elif strand == "-":
                        predict_outcome[chr_s][position] = position + args.window_length - pred_max_indices[i,0]
                    output = [chr,strand,position,sample,pred_max_indices[i,0],pred[i,0,pred_max_indices[i,0]]]
                    file.write("\t".join(map(str, output)) + "\n")

        
    output_file_path = args.prefix + "_correct.gtf"
    file_path = args.short_path + "/" + args.input + "_stringtie.gtf"
    corrected_transcripts = tss_correct(file_path,predict_outcome)
    write_gtf(corrected_transcripts,output_file_path)



