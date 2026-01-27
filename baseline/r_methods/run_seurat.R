#!/usr/bin/env Rscript
# Seurat CCA/RPCA batch correction
# Usage: Rscript run_seurat.R input.h5ad output.tsv [method: cca|rpca]

suppressPackageStartupMessages({
  library(Seurat)
  library(zellkonverter)
  library(SingleCellExperiment)
})

suppressPackageStartupMessages({
  library(future)
})

future::plan("sequential")
options(future.globals.maxSize = 8 * 1024^3)  # 8GiB


args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("Usage: Rscript run_seurat.R <input.h5ad> <output.tsv> [method: cca|rpca]")

input_path <- args[1]
output_path <- args[2]
method <- ifelse(length(args) >= 3, tolower(args[3]), "cca")
if (!(method %in% c("cca", "rpca"))) stop("method must be 'cca' or 'rpca'")

sce <- readH5AD(input_path)

if (!"BATCH" %in% colnames(colData(sce))) stop("Missing obs column: BATCH")
if (is.null(colnames(sce)) || length(colnames(sce)) == 0) stop("SCE has no cell names (colnames)")

assays_avail <- assayNames(sce)
cat("Available assays:", paste(assays_avail, collapse = ", "), "\n")

# Pick a matrix to treat as counts
counts_assay <- NULL
if ("counts" %in% assays_avail) {
  counts_assay <- "counts"
} else if ("X" %in% assays_avail) {
  counts_assay <- "X"
} else {
  stop(paste0("No usable assay found. Found: ", paste(assays_avail, collapse = ", ")))
}

mat <- assay(sce, counts_assay)
if (is.null(mat) || nrow(mat) == 0 || ncol(mat) == 0) stop(paste0("Assay '", counts_assay, "' is empty."))

# Build Seurat object directly from matrix (avoids requiring logcounts)
seurat_obj <- CreateSeuratObject(counts = mat)
seurat_obj$batch <- as.factor(colData(sce)$BATCH)

seurat_list <- SplitObject(seurat_obj, split.by = "batch")

# Normalize and HVGs
seurat_list <- lapply(seurat_list, function(x) {
  x <- NormalizeData(x, verbose = FALSE)
  x <- FindVariableFeatures(x, selection.method = "vst", nfeatures = 2000, verbose = FALSE)
  x
})

features <- SelectIntegrationFeatures(object.list = seurat_list, nfeatures = 2000)

# Find anchors
if (method == "rpca") {
  seurat_list <- lapply(seurat_list, function(x) {
    x <- ScaleData(x, features = features, verbose = FALSE)
    x <- RunPCA(x, features = features, verbose = FALSE)
    x
  })

  anchors <- FindIntegrationAnchors(
    object.list = seurat_list,
    anchor.features = features,
    reduction = "rpca",
    dims = 1:30,
    verbose = FALSE
  )
} else {
  anchors <- FindIntegrationAnchors(
    object.list = seurat_list,
    anchor.features = features,
    dims = 1:30,
    verbose = FALSE
  )
}

integrated <- IntegrateData(anchorset = anchors, dims = 1:30, verbose = FALSE)
DefaultAssay(integrated) <- "integrated"

integrated <- ScaleData(integrated, features = features, verbose = FALSE)
integrated <- RunPCA(integrated, npcs = 50, verbose = FALSE)

pca_embed <- Embeddings(integrated, reduction = "pca")

# Write embedding TSV (rownames are cell ids)
write.table(
  pca_embed,
  file = output_path,
  sep = "\t",
  quote = FALSE,
  col.names = NA
)

cat("Seurat", toupper(method), "completed:", output_path, "\n")
