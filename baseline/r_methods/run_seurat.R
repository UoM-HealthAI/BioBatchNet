#!/usr/bin/env Rscript
# Seurat CCA/RPCA batch correction
# Usage: Rscript run_seurat.R input.h5ad output.h5ad [method: cca|rpca]

suppressPackageStartupMessages({
    library(Seurat)
    library(zellkonverter)
    library(SingleCellExperiment)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("Usage: Rscript run_seurat.R <input.h5ad> <output.h5ad> [method: cca|rpca]")

input_path <- args[1]
output_path <- args[2]
method <- ifelse(length(args) >= 3, args[3], "cca")

# Read and convert
sce <- readH5AD(input_path)
seurat_obj <- as.Seurat(sce, counts = "X")
seurat_obj$batch <- colData(sce)$BATCH
seurat_list <- SplitObject(seurat_obj, split.by = "batch")

# Normalize and find variable features
seurat_list <- lapply(seurat_list, function(x) {
  x <- NormalizeData(x, verbose = FALSE)
  x <- FindVariableFeatures(x, selection.method = "vst", nfeatures = 2000, verbose = FALSE)
  x
})

features <- SelectIntegrationFeatures(object.list = seurat_list, nfeatures = 2000)

# Find anchors (RPCA needs PCA first)
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

# Integrate and get PCA
integrated <- IntegrateData(anchorset = anchors, dims = 1:30, verbose = FALSE)
DefaultAssay(integrated) <- "integrated"
integrated <- ScaleData(integrated, features = features, verbose = FALSE)
integrated <- RunPCA(integrated, npcs = 50, verbose = FALSE)


# Save
reducedDim(sce, paste0("X_seurat", method)) <- Embeddings(integrated, reduction = "pca")
writeH5AD(sce, output_path)
cat("Seurat", toupper(method), "completed:", output_path, "\n")
