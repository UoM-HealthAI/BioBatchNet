#!/usr/bin/env Rscript
# ComBat-seq batch correction
# Usage: Rscript run_combatseq.R input.h5ad output.tsv

suppressPackageStartupMessages({
  library(sva)
  library(SingleCellExperiment)
  library(zellkonverter)
  library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("Usage: Rscript run_combatseq.R <input.h5ad> <output.tsv>")

input_path <- args[1]
output_path <- args[2]

sce <- readH5AD(input_path)

if (!"BATCH" %in% colnames(colData(sce))) stop("Missing obs column: BATCH")
batch <- as.factor(colData(sce)$BATCH)

# Get counts matrix (genes x cells)
if ("counts" %in% assayNames(sce)) {
  counts <- assay(sce, "counts")
} else if ("X" %in% assayNames(sce)) {
  counts <- assay(sce, "X")
} else {
  stop("No assay named 'counts' or 'X' found.")
}

# Ensure dense matrix for ComBat_seq
if (inherits(counts, "sparseMatrix")) {
  counts <- as.matrix(counts)
}

# Run ComBat-seq (expects genes x cells)
cat("Running ComBat-seq...\n")
corrected <- ComBat_seq(counts, batch = batch)

# Normalize + log1p (column-wise, cells are columns)
lib <- colSums(corrected)
lib[lib == 0] <- 1
norm <- t(t(corrected) / lib) * 1e4
log_norm <- log1p(norm)

# PCA (need cells x genes for prcomp)
cat("Running PCA...\n")
pca_result <- prcomp(t(log_norm), center = TRUE, scale. = FALSE, rank. = 50)
emb <- pca_result$x[, 1:50]

# Set rownames as cell ids
rownames(emb) <- colnames(sce)

# Write embedding as tsv
write.table(
  emb,
  file = output_path,
  sep = "\t",
  quote = FALSE,
  col.names = NA
)

cat("ComBat-seq completed:", output_path, "\n")
