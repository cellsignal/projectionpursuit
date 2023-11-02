library(dplyr)
library(Seurat)
library(patchwork)

# Load the PBMC dataset
pbmc.data <- Read10X(data.dir = "~/Desktop/Collaborations/Colonna/EPP (2)/filtered_gene_bc_matrices/hg19/")
# Initialize the Seurat object with the raw (non-normalized data).
pbmc <- CreateSeuratObject(counts = pbmc.data, project = "pbmc3k", min.cells = 3, min.features = 200)
pbmc

pbmc[["percent.mt"]] <- PercentageFeatureSet(pbmc, pattern = "^MT-")
VlnPlot(pbmc, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), ncol = 3)

plot1 <- FeatureScatter(pbmc, feature1 = "nCount_RNA", feature2 = "percent.mt")
plot2 <- FeatureScatter(pbmc, feature1 = "nCount_RNA", feature2 = "nFeature_RNA")
plot1 + plot2

pbmc <- subset(pbmc, subset = nFeature_RNA > 200 & nFeature_RNA < 2500 & percent.mt < 5)
pbmc <- NormalizeData(pbmc, normalization.method = "LogNormalize", scale.factor = 10000)

pbmc <- FindVariableFeatures(pbmc, selection.method = "vst", nfeatures = 2000)

# Identify the 10 most highly variable genes
top10 <- head(VariableFeatures(pbmc), 10)

# plot variable features with and without labels
plot1 <- VariableFeaturePlot(pbmc)
plot2 <- LabelPoints(plot = plot1, points = top10, repel = TRUE)
plot1 + plot2

all.genes <- rownames(pbmc)
pbmc <- ScaleData(pbmc, features = all.genes)

pbmc <- RunPCA(pbmc, features = VariableFeatures(object = pbmc))

DimPlot(pbmc, reduction = "pca")

DimHeatmap(pbmc, dims = 1, cells = 500, balanced = TRUE)
pbmc <- FindNeighbors(pbmc, dims = 1:10)
pbmc <- FindClusters(pbmc, resolution = 0.5)

pbmc <- RunUMAP(pbmc, dims = 1:10)
# note that you can set `label = TRUE` or use the LabelClusters function to help label
# individual clusters
DimPlot(pbmc, reduction = "umap")

FeaturePlot(pbmc, features = c("MS4A1", "GNLY", "CD3E", "CD14", "FCER1A", "FCGR3A", "LYZ", "PPBP",
                               "CD8A"))

pbmc.markers <- FindAllMarkers(pbmc, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25)
pbmc.markers %>%
    group_by(cluster) %>%
    slice_max(n = 2, order_by = avg_log2FC)

pbmc.markers %>%
  group_by(cluster) %>%
  top_n(n = 10, wt = avg_log2FC) -> top10
DoHeatmap(pbmc, features = top10$gene) + NoLegend()


new.cluster.ids <- c("Naive CD4 T", "CD14+ Mono", "Memory CD4 T", "B", "CD8 T", "FCGR3A+ Mono",
                     "NK", "DC", "Platelet")
names(new.cluster.ids) <- levels(pbmc)
pbmc <- RenameIdents(pbmc, new.cluster.ids)
DimPlot(pbmc, reduction = "umap", label = TRUE, pt.size = 0.5) + NoLegend()



pca <- pbmc@reductions$pca@cell.embeddings
pca <- as.data.frame(pca)

pca_10 <- pca[,1:20]


metadata <- pbmc@meta.data
metadata < as.data.frame(metadata)
metadata$cluster_name <- pbmc@active.ident

pca_10$cluster_id <- metadata$seurat_clusters

write.csv(pca_10, "~/Desktop/Collaborations/Colonna/EPP (2)/pbmc_pca_num_20.csv", row.names = FALSE)
table(Idents(pbmc))

write.csv(metadata, "~/Desktop/Collaborations/Colonna/pbmc_metadata.csv", row.names = FALSE)


projection_pursuit_pca2_csv <- read_csv("~/Desktop/Collaborations/Colonna/EPP (2)/dml/projection_pursuit_pbmc_pca_num_min10.csv.csv")
ground_truth_pca2_csv <- read_csv("~/Desktop/Collaborations/Colonna/EPP (2)/dml/ground_truth_pbmc_pca_num_min10.csv.csv")


pbmc@meta.data$projection_pursuit_cluster <- projection_pursuit_pca2_csv$cluster_id
#change active ident (save old active ident)

pbmc@meta.data$named_cluster <- pbmc@active.ident

pbmc <- SetIdent(pbmc, value = "projection_pursuit_cluster")
pbmc <- SetIdent(pbmc, value = "seurat_clusters")


DimPlot(pbmc)

levels(pbmc)
new.cluster.ids <- c("CD4/CD8", "CD14+ Mono", "B", "FCGR3A+ Mono", "NK")
names(new.cluster.ids) <- levels(pbmc)
pbmc <- RenameIdents(pbmc, "3" = "CD4/CD8", "2" = "CD14+ Mono", "4" = "B", "1" = "FCGR3A+ Mono", "0" = "NK")
DimPlot(pbmc, reduction = "umap", pt.size = 0.5)

pbmc@meta.data$pp_named_cluster <- wt1@active.ident

#SUPERVISED UMAPS

library(uwot)
df <- as.data.frame(t(as.matrix(pbmc@assays$RNA@data)))

umap_result <- umap(df, y = pbmc@meta.data$seurat_clusters)
umap_result <- as.data.frame(umap_result)
umap_result$named_cluster <- pbmc@meta.data$seurat_clusters
ggplot(umap_result, aes(x = V1, y = V2, color = named_cluster)) + geom_point()



umap_result <- umap(df, y = pbmc@meta.data$projection_pursuit_cluster)
umap_result <- as.data.frame(umap_result)
umap_result$named_cluster <- as.character(pbmc@meta.data$projection_pursuit_cluster)
ggplot(umap_result, aes(x = V1, y = V2, color = named_cluster)) + geom_point()

umap_result <- umap(df)
umap_result <- as.data.frame(umap_result)
umap_result$named_cluster <- pbmc@meta.data$seurat_clusters
ggplot(umap_result, aes(x = V1, y = V2, color = named_cluster)) + geom_point()


