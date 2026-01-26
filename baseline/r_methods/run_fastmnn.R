#!/usr/bin/env Rscript
# FastMNN batch correction
# Usage: Rscript run_fastmnn.R input.h5ad output.h5ad

suppressPackageStartupMessages({
  library(batchelor)
  library(SingleCellExperiment)
  library(zellkonverter)
  library(scater)   # for logNormCounts
  library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("Usage: Rscript run_fastmnn.R <input.h5ad> <output.h5ad>")

input_path <- args[1]
output_path <- args[2]

sce <- readH5AD(input_path)

if (!"BATCH" %in% colnames(colData(sce))) stop("Missing obs column: BATCH")
batch <- as.factor(colData(sce)$BATCH)

# Ensure logcounts exists and is truly log-normalized
if (!"logcounts" %in% assayNames(sce)) {
  if (!"counts" %in% assayNames(sce)) assay(sce, "counts") <- assay(sce, "X")  
  sce <- logNormCounts(sce)  # creates logcounts
}

corrected <- fastMNN(sce, batch = batch)

emb <- reducedDim(corrected, "corrected")
emb <- emb[colnames(sce), , drop = FALSE]

reducedDim(sce, "X_fastmnn") <- emb

writeH5AD(sce, output_path)
cat("fastMNN completed:", output_path, "\n")
