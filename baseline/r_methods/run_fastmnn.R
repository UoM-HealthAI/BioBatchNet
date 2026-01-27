#!/usr/bin/env Rscript
# FastMNN batch correction
# Usage: Rscript run_fastmnn.R input.h5ad output.tsv

suppressPackageStartupMessages({
  library(batchelor)
  library(SingleCellExperiment)
  library(zellkonverter)
  library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("Usage: Rscript run_fastmnn.R <input.h5ad> <output.tsv>")

input_path <- args[1]
output_path <- args[2]

sce <- readH5AD(input_path)

if (!"BATCH" %in% colnames(colData(sce))) stop("Missing obs column: BATCH")
batch <- as.factor(colData(sce)$BATCH)

# Ensure logcounts exists
if (!"logcounts" %in% assayNames(sce)) {
  # Prefer 'counts', else fall back to 'X' if present
  if ("counts" %in% assayNames(sce)) {
    counts <- assay(sce, "counts")
  } else if ("X" %in% assayNames(sce)) {
    counts <- assay(sce, "X")
    assay(sce, "counts") <- counts
  } else {
    stop("No assay named 'counts' or 'X' found. Please write counts to the input h5ad.")
  }

  # library-size normalize per cell (per column), then log1p
  # counts is usually (genes x cells)
  lib <- Matrix::colSums(counts)
  lib[lib == 0] <- 1
  scale_factor <- 1e4

  # Correct column-wise scaling for sparse matrices:
  norm <- t(t(counts) / lib) * scale_factor
  assay(sce, "logcounts") <- log1p(norm)
}

corrected <- fastMNN(sce, batch = batch)

emb <- reducedDim(corrected, "corrected")

# Align embedding rows to original cell order
# reducedDim(corrected) should have rownames as cell ids; enforce and reorder
if (is.null(rownames(emb))) {
  rownames(emb) <- colnames(sce)
}
emb <- emb[colnames(sce), , drop = FALSE]

# Write embedding only (tsv). Row names are cell ids.
# Use col.names=NA to keep rownames in first column.
write.table(
  emb,
  file = output_path,
  sep = "\t",
  quote = FALSE,
  col.names = NA
)

cat("fastMNN completed:", output_path, "\n")
