#!/usr/bin/python
import argparse
import sys
import re
import random
import string
import time
import subprocess
import logging
import os

parser = argparse.ArgumentParser(description='TEIRI_merge')
parser.add_argument('-i', '--gtf_list', help="A text file with a list of GTF files (required)")
parser.add_argument('-r', '--reference_gtf', help="Reference genome annotation in GTF format (required)")
parser.add_argument('--TE_anno', help="TE annotation in BED format (required)")
parser.add_argument( '--tss_window', default=50, type=int, help='The window size used to calculate the weight score (default: 50)')
parser.add_argument('--max_tss', default=1, type=int, help='Maximum number of TSS picked per first exon (default: 1)')
parser.add_argument('--min_NGS_samples', default=20, type=float, help='The min NGS ratio supporting the first exon (default: 10)')
parser.add_argument('--max_transcripts', default=10, type=int, help='The max counts of transcripts for a TE-initiated RNA (default: 10)')
parser.add_argument('--trunctated_exclude', default=True, help='Truncated transcripts were excluded, as they may represent fragments of the full-length transcript (default: True)')
parser.add_argument('--min_transcript_length', default=200, type=float, help='The min transcript length (default: 200)')
parser.add_argument('--threshold', default=1, type=float, help='The threshold of weight score for TSS (default: 1)')
parser.add_argument('-p', '--prefix', default='TEIRI', help="Prefix for output file (default: TEIRI)")

args = parser.parse_args()


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
    return transcript


def process_bed12(bed12_path):
    transcript = {}

    with open(bed12_path) as nanopore:
        for eachline in nanopore:
            if not eachline.strip().startswith('#'):
                temp = eachline.strip().split("\t")
                id = "\t".join([temp[0],str(temp[5]),temp[3]])
                exon_length = temp[10].split(",")
                exon_start = temp[11].split(",")
                transcript[id] = []
                for i in range(0,int(temp[9])):
                    transcript[id].append(int(temp[1]) + int(exon_start[i]) + 1)
                    transcript[id].append(int(temp[1]) + int(exon_start[i]) + int(exon_length[i]))
                    

    return transcript

def tss_correct(transcipt,TGS_weight,max_tss,tss_window, threshold):
    raw_tss = {}
    corrected_tss = {}

    for gtf in transcipt:
        if gtf != "reference":
            for transciptID in transcipt[gtf]:
                id = transciptID.split("\t")
                exons=transcipt[gtf][transciptID]
                exons.sort() 
                num_exons=int(len(exons)/2)
                chr = "\t".join([id[0],id[1]])
                if chr not in raw_tss:
                    raw_tss[chr] = {}
                    corrected_tss[chr] = {}
                if num_exons>=2 :
                    if id[1]=="+":
                        exon1_end = exons[1]
                        tss = exons[0]
                    elif id[1]=="-":
                        exon1_end = exons[-2]
                        tss = exons[-1] 
                    if exon1_end not in raw_tss[chr]:
                        raw_tss[chr][exon1_end]={}
                        corrected_tss[chr][exon1_end]=[]
                    if tss not in raw_tss[chr][exon1_end]:
                        raw_tss[chr][exon1_end][tss]=[[],0]
                    if gtf !="nanopore" : 
                        raw_tss[chr][exon1_end][tss][0].append(gtf)
                    else:
                        raw_tss[chr][exon1_end][tss][1]+=1





    for chr in raw_tss:
        strand = chr.split("\t")[1]
        for exon1_end in raw_tss[chr]:
            tss_count = {}
            best_tss = {}
            for tss in raw_tss[chr][exon1_end]:
                tss_count[tss] =  raw_tss[chr][exon1_end][tss][1]* TGS_weight + len(list(set(raw_tss[chr][exon1_end][tss][0])))
            tss_range=list(tss_count.keys()) 

            for k in range(0,int(max_tss)):
                if len(tss_range) >= 1 :
                    best_tss[str(k)] = [0,0,0]
                    weighted_score = dict.fromkeys(tss_range, 0) 
                    if strand == "+" :
                        sorted_tss=sorted(tss_range,reverse=True)
                    elif strand == "-":
                        sorted_tss=sorted(tss_range)
                    for tss_1 in sorted_tss:
                        for tss_2 in sorted_tss:
                            if tss_1 == tss_2:
                                weighted_score[tss_1] += tss_count[tss_1]
                            elif (abs(tss_1 - tss_2) < tss_window) :
                                weighted_score[tss_1] += ((tss_window - abs(tss_1 - tss_2))/float(tss_window * tss_count[tss_2]))
                        if weighted_score[tss_1] > best_tss[str(k)][2] :
                            best_tss[str(k)] = [tss_1,tss_count[tss_1],weighted_score[tss_1] ]

                    
                    temp = sorted_tss
                    for tss_1 in temp:
                        if abs(tss_1 - best_tss[str(k)][0]) < tss_window:
                            tss_range.remove(tss_1)
                    if k==0 and best_tss[str(k)][2] > threshold:
                        corrected_tss[chr][exon1_end].append(best_tss[str(k)][0])
                    elif best_tss[str(k)][2] > threshold and best_tss[str(k)][2]/best_tss[str(0)][2] > 0.5:
                        corrected_tss[chr][exon1_end].append(best_tss[str(k)][0])
    
    return corrected_tss



def exon_extract(transcipt,corrected_tss):
    output_exons = {}
    output_exons["first_exon"] = {}
    output_exons["inner_exon"] = {}
    output_exons["final_exon"] = {}
    output_exons["single_exon"] = {}

    for gtf in transcipt:
        for transciptID in transcipt[gtf]:
            id = transciptID.split("\t")
            exons=transcipt[gtf][transciptID]
            exons.sort() 
            num_exons=int(len(exons)/2)
            chr = "\t".join([id[0],id[1]])
            if num_exons>=2 :
                if id[1]=="+":
                    if exons[1] in corrected_tss[chr]:
                        tss_set = corrected_tss[chr][exons[1]]
                        corrected_status = True
                    else:
                        corrected_status = False
                        tss_set = []
                    exons[-1] = exons[-2] + 20
                elif id[1]=="-":
                    if exons[-2] in corrected_tss[chr]:
                        tss_set = corrected_tss[chr][exons[-2]]
                        corrected_status = True
                    else:
                        corrected_status = False
                        tss_set = []
                    exons[0] = exons[1] - 20
                for i in range(num_exons):
                    exon_type = "inner_exon"
                    if id[1]=="+":
                        if i==0:
                            exon_type = "first_exon"
                        elif i==(num_exons-1):
                            exon_type = "final_exon"
                    elif id[1]=="-":
                        if i==0:
                            exon_type = "final_exon"
                        elif i==(num_exons-1):
                            exon_type = "first_exon"

                    if exon_type == "first_exon" : 
                        if corrected_status :
                            for tss in tss_set:
                                if id[1]=="+":
                                    exons[0] = tss
                                elif id[1]=="-":
                                    exons[-1] = tss

                                exon_temp="\t".join([str(exons[2*i]-1),str(exons[2*i+1])])
                                if chr not in output_exons[exon_type]:
                                    output_exons[exon_type][chr]={}
                                if exon_temp not in output_exons[exon_type][chr]:
                                    output_exons[exon_type][chr][exon_temp] = [[],0,"no_ref"]
                                if gtf == "reference":
                                    output_exons[exon_type][chr][exon_temp][2]="ref"
                                elif gtf == "nanopore":
                                    output_exons[exon_type][chr][exon_temp][1]+=1
                                else:
                                    output_exons[exon_type][chr][exon_temp][0].append(gtf)
                    else:
                        exon_temp="\t".join([str(exons[2*i]-1),str(exons[2*i+1])])
                        if chr not in output_exons[exon_type]:
                            output_exons[exon_type][chr]={}
                        if exon_temp not in output_exons[exon_type][chr]:
                            output_exons[exon_type][chr][exon_temp] = [[],0,"no_ref"]
                        if gtf == "reference":
                            output_exons[exon_type][chr][exon_temp][2]="ref"
                        elif gtf == "nanopore":
                            output_exons[exon_type][chr][exon_temp][1]+=1
                        else:
                            output_exons[exon_type][chr][exon_temp][0].append(gtf)
    
    return(output_exons)








def generate_random_string(length=8):
    logging.info("Generating random string for temp file name...")
    letters = string.ascii_lowercase
    random_string = ''.join(random.choice(letters) for i in range(length))
    logging.info("Random string generation completed.")
    return random_string

def write_exon_bed_file(results, output_file,min_samples,min_TGS_reads):
    logging.info(f'Writing BED file to: {output_file}')
    with open(output_file, "w") as f_out:
        for chr in results:
            for exon_temp in results[chr]:
                nanopore_reads = results[chr][exon_temp][1]
                ref = results[chr][exon_temp][2]

                samples = list(set(results[chr][exon_temp][0]))
                chr_ = chr.split("\t")
                result = "\t".join([chr_[0],exon_temp,ref,str(len(samples)),chr_[1]]) 
                if len(samples) >= min_samples or nanopore_reads >= min_TGS_reads   :
                    f_out.write(result + "\n")
                #elif ref=="ref" and (len(samples) + nanopore_reads) >= 1:
                #    f_out.write(result + "\n")
    logging.info('BED file writing completed.')

def exon_filter(temp_file):
    first = temp_file  + "_first_exon.bed"
    inner = temp_file  + "_inner_exon.bed" 
    final = temp_file  + "_final_exon.bed"

    cmd = "cat " + inner + " " + final + " > " + temp_file +  "_inner_final.bed"
    subprocess.check_call(cmd, shell=True)
    cmd = "bedtools subtract -a " + first + " -b " + temp_file +  "_inner_final.bed  -s -A|cut -f 1-6|sort -k1,1 -k2,2n|uniq > " + temp_file +  "_final_exon1.bed"
    subprocess.check_call(cmd, shell=True)

def write_bed_file(corrected_tss, output_file):
    logging.info(f'Writing BED file to: {output_file}')
    with open(output_file, "w") as f_out:
        for tss in corrected_tss:
            f_out.write("\t".join(map(str, tss)) + "\n")
    logging.info('BED file writing completed.')


def te_tss_extract(tss_bed,TE_anno,temp_file):
    cmd = "bedtools intersect -wo -a " +  tss_bed + " -b " + TE_anno +  " > " + temp_file +  "_intersect.bed"
    subprocess.check_call(cmd, shell=True)
    intersect_file = temp_file +  "_intersect.bed"
    with open(intersect_file) as file:
        for eachline in file:
            temp = eachline.strip().split("\t")
            if temp[5] == "+" and int(temp[1]) < int(temp[8]) and int(temp[1]) > int(temp[7])  :
                yield temp[:6]
            elif temp[5] == "-" and int(temp[2]) < int(temp[8]) and int(temp[2]) > int(temp[7])  :
                yield temp[:6]


def process_bed(bed_path):
    te_tss = {}

    with open(bed_path) as bed:
        for eachline in bed:
            if not eachline.strip().startswith('#') :
                temp = eachline.strip().split("\t")
                chr="\t".join([temp[0],temp[5]])
                if chr not in te_tss:
                    te_tss[chr] = {}
                id=[str(int(temp[1])+1),str(temp[2])]
                if temp[5]=="-":
                    id.reverse()
                if id[1] not in te_tss[chr]:
                    te_tss[chr][id[1]]=[str(id[0])]
                else:
                    te_tss[chr][id[1]].append(str(id[0]))
    
    return te_tss

def splicing_sites_extract(reference_gtf):
    ss1 = {}
    ss2 = {}
    
    for transcriptID in reference_gtf:
        id = transcriptID.split("\t")
        exons=reference_gtf[transcriptID]
        exons.sort() 
        num_exons=int(len(exons)/2)
        chr = "\t".join([id[0],id[1]])
        if  num_exons>=2 :
            if chr not in ss1:
                ss1[chr]={}
                ss2[chr]={}
            for i in range(0,num_exons-1):
                ss1_temp = str(exons[2*i+2])
                ss2_temp = str(exons[2*i+1])
                if ss1_temp not in ss1[chr]:
                    ss1[chr][ss1_temp]=[transcriptID]
                else:
                    ss1[chr][ss1_temp].append(transcriptID)
                if ss2_temp not in ss2[chr]:
                    ss2[chr][ss2_temp]=[transcriptID]
                else:
                    ss2[chr][ss2_temp].append(transcriptID)
    return ss1,ss2

def ref_exons_generate(reference_gtf,ss,i,chr,strand,ss1,ss2):
    reference_transcripts=[]
    if strand == "+":
        if ss in ss2[chr] and i % 2==1:
            reference_transcripts = ss2[chr][ss]
        elif ss in ss1[chr] and i % 2==0:
            reference_transcripts = ss1[chr][ss]
        if len(reference_transcripts)>=1:
            for ID in reference_transcripts:
                ref_exons = reference_gtf[ID]
                ref_exons.sort()
                ref_exons = ref_exons[1:]
                ref_exons = list(map(str, ref_exons))
                ref_exons = ref_exons[ref_exons.index(ss):]
                yield ID,ref_exons
        else:
            yield "None","None"

    elif strand == "-":
        if ss in ss1[chr] and i % 2==1:
            reference_transcripts = ss1[chr][ss]
        elif ss in ss2[chr] and i % 2==0:
            reference_transcripts = ss2[chr][ss]
        if len(reference_transcripts)>=1:
            for ID in reference_transcripts:
                ref_exons = reference_gtf[ID]
                ref_exons.sort(reverse=True)
                ref_exons = ref_exons[1:]
                ref_exons = list(map(str, ref_exons))
                ref_exons = ref_exons[ref_exons.index(ss):]
                yield ID,ref_exons
        else:
            yield "None","None"





def transcript_correct(transcript,reference_gtf,te_tss,ss1,ss2):
    corrected_transcript={}
    for transcriptID in list(transcript.keys()):
        id = transcriptID.split("\t")
        exons=transcript[transcriptID]
        num_exons=int(len(exons)/2)
        chr = "\t".join([id[0],id[1]])
        if id[1]=="+":
            exons.sort() 
        elif id[1]=="-":
            exons.sort(reverse=True)
        if chr in te_tss:
            if  (num_exons>=2) and (str(exons[1]) in te_tss[chr]):
                new_exons = ["\t".join(te_tss[chr][str(exons[1])])]
                correct = 0
                if chr in ss1:
                    for i in range(1,2*num_exons-1):
                        for refID,ref_exons in ref_exons_generate(reference_gtf,str(exons[i]),i,chr,id[1],ss1,ss2):
                            if refID !="None":
                                new_transcriptID = "_".join([transcriptID , refID ])
                                corrected_transcript[new_transcriptID]= new_exons + ref_exons
                                correct = 1
                        if correct == 1:
                            break
                        else:
                            new_exons.append(str(exons[i]))
                else:
                    for i in range(1,2*num_exons-1):
                        new_exons.append(str(exons[i]))
                if correct != 1 : 
                    new_exons.append(str(exons[-1]))
                    corrected_transcript[transcriptID] = new_exons
    return corrected_transcript

def transcript_merge(corrected_transcript,TGS_weight,illumina_threshold,nanopore_threshold,trunctated_exclude,max_transcripts,min_transcript_length):
    mergeded_transcript = {}
    for gtf in corrected_transcript:
        for transcriptID in list(corrected_transcript[gtf].keys()):
            id = transcriptID.split("\t")
            exons=corrected_transcript[gtf][transcriptID]
            ss_all = "\t".join(exons[1:-1])
            num_exons=int(len(exons)/2)
            chr = "\t".join([id[0],id[1]])
            intron1 = "\t".join([exons[1],exons[2]])
            
            for tss_ in exons[0].split("\t"):
                
                if chr not in mergeded_transcript:
                    mergeded_transcript[chr] = {}
                if tss_ not in mergeded_transcript[chr]:
                    mergeded_transcript[chr][tss_]={}
                if intron1 not in mergeded_transcript[chr][tss_]:
                    mergeded_transcript[chr][tss_][intron1]={}
                if ss_all not in mergeded_transcript[chr][tss_][intron1]:
                    if gtf == "ref" :
                        mergeded_transcript[chr][tss_][intron1][ss_all] = [[],1000,int(tss_),int(exons[-1])]
                    elif gtf == "nanopore" :
                        mergeded_transcript[chr][tss_][intron1][ss_all] = [[],1,int(tss_),int(exons[-1])]
                    else:
                        mergeded_transcript[chr][tss_][intron1][ss_all] = [[gtf],0,int(tss_),int(exons[-1])]
                else:
                    if gtf == "ref" :
                        mergeded_transcript[chr][tss_][intron1][ss_all][1] += 1000
                    elif gtf == "nanopore" :
                        mergeded_transcript[chr][tss_][intron1][ss_all][1] += 1
                    else:
                        mergeded_transcript[chr][tss_][intron1][ss_all][0].append(gtf)
                    if (id[1]=="+") and (int(exons[-1]) < mergeded_transcript[chr][tss_][intron1][ss_all][3]):
                        mergeded_transcript[chr][tss_][intron1][ss_all][3] = int(exons[-1])
                    if (id[1]=="-") and (int(exons[-1]) > mergeded_transcript[chr][tss_][intron1][ss_all][3]):
                        mergeded_transcript[chr][tss_][intron1][ss_all][3] = int(exons[-1])


    for chr in mergeded_transcript:
        for tss_ in list(mergeded_transcript[chr].keys()):
            for intron1 in list(mergeded_transcript[chr][tss_].keys()):
                for ss_all in list(mergeded_transcript[chr][tss_][intron1].keys()):
                    exons = [mergeded_transcript[chr][tss_][intron1][ss_all][2]] + ss_all.split("\t") + [mergeded_transcript[chr][tss_][intron1][ss_all][3]]
                    transcript_length = float( transcript_length_calculate(exons))
                    nanopore_reads = mergeded_transcript[chr][tss_][intron1][ss_all][1]
                    illumina_score = len(list(set(mergeded_transcript[chr][tss_][intron1][ss_all][0]))) 
                    if args.corrected_bed12:
                        nanopore_score = nanopore_reads*TGS_weight*transcript_length/1000
                        weight_score = illumina_score + nanopore_score
                    else:
                        weight_score = illumina_score
                    if transcript_length < min_transcript_length :
                        del mergeded_transcript[chr][tss_][intron1][ss_all]
                    else:
                        mergeded_transcript[chr][tss_][intron1][ss_all][0]=float(weight_score)

    if trunctated_exclude:
        for chr in mergeded_transcript:
            for tss_ in list(mergeded_transcript[chr].keys()):
                for intron1 in list(mergeded_transcript[chr][tss_].keys()):
                    for ss_all in list(mergeded_transcript[chr][tss_][intron1].keys()):
                        for ss_all_ in list(mergeded_transcript[chr][tss_][intron1].keys()):
                            if (ss_all in ss_all_) and ss_all.split("\t")[0]==ss_all_.split("\t")[0]  and mergeded_transcript[chr][tss_][intron1][ss_all][0] < mergeded_transcript[chr][tss_][intron1][ss_all_][0]:
                                del mergeded_transcript[chr][tss_][intron1][ss_all]
                                break

    final_transcript = {}
    for chr in mergeded_transcript:
        final_transcript[chr] = []
        for tss_ in mergeded_transcript[chr]:
            for intron1 in list(mergeded_transcript[chr][tss_].keys()):
                score = []
                for ss_all in mergeded_transcript[chr][tss_][intron1]:
                    score.append(float( mergeded_transcript[chr][tss_][intron1][ss_all][0]))
                score.sort(reverse=True)
                max_score = set(score[0:max_transcripts])
                for ss_all in mergeded_transcript[chr][tss_][intron1]:
                    exons = [tss_] + ss_all.split("\t") + [mergeded_transcript[chr][tss_][intron1][ss_all][3]]
                    weight_score = mergeded_transcript[chr][tss_][intron1][ss_all][0]
                    if weight_score in max_score :
                        new_transcript = "\t".join([str(x) for x in exons])
                        final_transcript[chr].append(new_transcript)

    return final_transcript



def transcript_length_calculate(exons_temp):
    exons_temp = list(map(int, exons_temp))
    exons_temp.sort()
    num_exons=int(len(exons_temp)/2)
    transcript_length_temp = 0
    if  num_exons>=2 :
        for i in range(0,num_exons):
            transcript_length_temp += (int(exons_temp[2*i+1])-int(exons_temp[2*i]))
    return (transcript_length_temp)



def write_exon(exons,transcriptid,chr_,strand):
    exons = list(map(int, exons))
    num_exons=int(len(exons)/2)
    exons.sort()
    exons = list(map(str, exons))
    stringreturn=[chr_, "TE_RNA","transcript" , exons[0] ,exons[-1],".", strand,".",str("transcript_id \"teRNA_"+ str(transcriptid) + "\";") ]
    stringreturn = "\t".join([str(x) for x in stringreturn])
    yield stringreturn
    for i in range(0,num_exons):
        stringreturn=[chr_, "TE_RNA","exon" , exons[2*i] ,exons[2*i+1],".", strand,".",str("transcript_id \"teRNA_"+ str(transcriptid) + "\";") ]
        stringreturn = "\t".join([str(x) for x in stringreturn])
        yield stringreturn

def write_gtf(mergeded_transcripts,output_file):
    transcriptid = 0

    with open(output_file, "w") as f_out:
        for chr in mergeded_transcripts:
            strand= chr.split("\t")[1]
            chr_= chr.split("\t")[0]
            for id in mergeded_transcripts[chr]:
                transcriptid += 1
                exons = id.split("\t")
                for stringreturn in write_exon(exons,transcriptid,chr_,strand):
                    f_out.write(stringreturn + "\n")



transcripts = {}
extracted_exons = {}

gtf_list = []
with open(args.gtf_list) as files:
    for each in files:
        gtf_list.append(each.strip())
files.close()  

for gtf in gtf_list:
    transcripts[gtf] = process_gtf(gtf)

if args.reference_gtf:
    transcripts["reference"] = process_gtf(args.reference_gtf)


if args.corrected_bed12:
    transcripts["nanopore"] = process_bed12(args.corrected_bed12)

corrected_tss = tss_correct(transcripts,args.TGS_weight,args.max_tss,args.tss_window, args.threshold)
output_exons = exon_extract(transcripts,corrected_tss)
temp_file = generate_random_string()

min_samples = args.min_NGS_samples 


for exon_type in output_exons:
    output_file = temp_file + "_" + exon_type + ".bed"
    write_exon_bed_file(output_exons[exon_type],output_file,min_samples,args.min_TGS_reads)
    

exon_filter(temp_file)

filter_exon1 = temp_file  + "_final_exon1.bed"


output_file = args.prefix + "_te_tss.bed"
write_bed_file(te_tss_extract(filter_exon1,args.TE_anno,temp_file),output_file)


cmd = "rm " +  temp_file + "*"
subprocess.check_call(cmd, shell=True)


te_tss_bed = process_bed(output_file)

ss1,ss2 = splicing_sites_extract(transcripts["reference"])

corrected_transcripts = {}

for gtf in transcripts:
    corrected_transcripts[gtf] = transcript_correct(transcripts[gtf],transcripts["reference"],te_tss_bed,ss1,ss2)
    print (gtf+"_corrected")

min_samples = args.min_NGS_samples 

mergeded_transcripts=transcript_merge(corrected_transcripts,args.TGS_weight,min_samples,args.min_TGS_reads,args.trunctated_exclude,args.max_transcripts,args.min_transcript_length)


output_file = args.prefix + "_TE.gtf"
write_gtf(mergeded_transcripts,output_file)

