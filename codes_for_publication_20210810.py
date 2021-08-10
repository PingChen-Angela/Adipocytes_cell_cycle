# Author: Ping Chen

import sys
from multi_dim_reduction import *
from file_process import *
from sample_cluster import *
from sample_filter import *
from vis_tools import *
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from IPython.display import Image
import re
from matplotlib_venn import venn2
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
pandas2ri.activate()
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import dendrogram, linkage

rpkms_mtx_file = 'inputs/adipocyte_bulk_rpkms.txt'
sample_annot_file = 'inputs/sample_groups.txt'  # set1_sample_annot
outputDIR = 'results'

rpkms_mtx = pd.read_table(rpkms_mtx_file, header=0, index_col=0, sep="\t")
sample_groups = pd.read_table(sample_annot_file, header=0, index_col=None, sep="\t")
sample_groups.index = sample_groups['sampleName'].to_list()

# Cell cycle genes
candidate_genes = ['CDKN1B', 'RB1', 'E2F4', 'RBL2', 'TFDP2', 'CCNG2', 'HEY2', 
                   'MCM2', 'CDK2', 'CCNE1', 'E2F3', 'PCNA', 'SP1', 'POLA1', 
                   'HIST1H2AC', 'ROCK1', 'RFC3', 'KAT2B', 'MSH3', 'PRIM1', 'TYMSOS', 
                   'CDKN1A', 'SRSF9', 'CCNA2', 'CDC20', 'CDC25B', 'MYBL2', 'FOXM1',
                   'ESPL1', 'PLK1', 'AURKA', 'AURKB', 'CENPA', 'KIF20A','CDK1','BUB1']

gid = pd.Series({gene.split('|')[0]: gene for gene in rpkms_mtx.index if gene.split('|')[0] in candidate_genes})
curr_expr = rpkms_mtx.loc[gid.values]
curr_expr.index = gid.index
curr_expr = curr_expr.loc[candidate_genes].fillna(0)
curr_expr.columns = sample_annot[curr_expr.columns]
sig_gene_expr = np.log2(curr_expr+1)
sample_cluster_df = sample_groups.copy()
sample_label = sample_cluster_df.loc[sig_gene_expr.columns]['sampleCluster']
sig_gene_expr.index = [gname.split('|')[0] for gname in sig_gene_expr.index]
masked_genes = list(sig_gene_expr.loc[sig_gene_expr.max(axis=1) < 1].index)
new_sig_gene_expr = sig_gene_expr.copy()
new_sig_gene_expr.loc[masked_genes] = np.nan

heatmap_gene_expr(new_sig_gene_expr, clust_method='ward', sample_label=sample_label, 
                  gene_level=None, row_cluster=False, yticklabels=True, xticklabels=True,
                  col_cluster=False, z_score=0, outdir=outputDIR, 
                  fig_width=6, fig_height=12, font_scale=1, filename="Figure1C_heatmap_cell_cycle_genes_zscore_bwr",
                  color_palette=['skyblue','salmon','red'],level1_name='Groups',
                  legend_x=0.7, legend_y=3, legend_y2=0.6, cbar=[1.1, .13, .05, .18],
                  ordered_sample_names=sig_gene_expr.columns, cmap_str='bwr')


# Differentially expressed genes
sigGenes_down = pd.read_table('%s/DEGs/dn/sigGeneStats.csv' %(outputDIR), index_col=0, header=0)
sigGenes_up = pd.read_table('%s/DEGs/up/sigGeneStats.csv' %(outputDIR), index_col=0, header=0)
sigGenes = list(set(list(sigGenes_down.index) + list(sigGenes_up.index)))

sig_gene_expr = np.log2(rpkms_mtx.loc[sigGenes]+1)
sample_cluster_df = sample_groups.copy()
sample_label = sample_cluster_df.loc[sig_gene_expr.columns]['sampleCluster']
sig_gene_expr.index = [gname.split('|')[0] for gname in sig_gene_expr.index]

heatmap_gene_expr(sig_gene_expr, clust_method='ward', sample_label=sample_label, 
                  gene_level=None, row_cluster=True, yticklabels=True, xticklabels=True,
                  col_cluster=False, z_score=0, outdir='%s/DEGs' %(outputDIR), 
                  fig_width=5, fig_height=18, font_scale=0.75, filename="heatmap_DEGs_zscore",
                  color_palette=['skyblue','salmon','red'],level1_name='Groups',
                  legend_x=0.7, legend_y=5, legend_y2=0.6, cbar=[1.1, .13, .03, .12],
                  ordered_sample_names=sig_gene_expr.columns)

k_clusters = kmeans(sig_gene_expr.T, 3)
k_clusters = pd.DataFrame(k_clusters)
k_clusters.index = k_clusters['sampleName']

comp = principle_component_analysis(rpkms_mtx.T, list(sigGenes), 
                                    n_comp=3, 
                                    annot=k_clusters.loc[rpkms_mtx.columns]['sampleCluster'],
                                    annoGE=None, log=True, size=500,
                                    pcPlot=True, pcPlotType='normal',
                                    markerPlot=False, with_mean=True, with_std=False,
                                    add_sample_label=False, 
                                    color_palette={'cluster1':colors['red'],
                                                   'cluster2':colors['skyblue'],'cluster3':colors['green']},
                                    figsize=(5,5), outdir=outputDIR, filename1='pca_color_by_clusters')

Z = linkage(sig_gene_expr.T, 'complete')
plt.figure(figsize=(8, 5))
plt.title('Hierarchical Clustering: Set1')
plt.xlabel('sample name')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=15,# font size for the x axis labels
    labels=list(sig_gene_expr.columns)
)
plt.tight_layout()
plt.savefig('%s/HC_plot.pdf' %(outputDIR), dpi=300)

# correlation
gene_factor_names = [gene for gene in rpkms_mtx.index if gene.split('|')[0] in ['CCND1','CDKN1A']]
annot_all = np.log2(rpkms_mtx.loc[gene_factor_names][sample_groups.index].T+1)
annot_all = pd.concat([sample_groups, annot_all], axis=1)
curr_expr_all = rpkms_mtx[annot_all.index].copy()

results = defaultdict(list)
for factor in gene_factor_names:
    print(factor)
    for gene in curr_expr_all.index:
        corr, pval = spearmanr(curr_expr_all.loc[gene].values, annot_all[factor].values, nan_policy='omit')
        if pval > 0.05: continue
        results[factor].append([gene, corr, pval])
        
with pd.ExcelWriter('%s/factor_associated_genes.xlsx' %(outputDIR)) as writer:
    for factor in results.keys():
        df = pd.DataFrame(results[factor], columns=['GeneName','Correlation','Pvalue']).sort_values(by='Pvalue', ascending=True)
        df.to_excel(writer, sheet_name=factor.split('|')[0], index=False)
