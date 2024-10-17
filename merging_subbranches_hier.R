library(monocle)
library(dplyr)
library(ggplot2)
library(dendextend)
library(cluster)

cds <- readRDS("../broadQRS_ddrtree_cds_reordered19.rds")

Z <- cds@reducedDimS
Y <- cds@reducedDimK
branch <- as.numeric(cds$State)

results <- tibble(
  Z1 = Z[1, ],
  Z2 = Z[2, ],
  branch = branch,
  Pseudotime = cds$Pseudotime,
  filename = cds$filename)

# Read in original data with other meta-data
bidmc_af_subset <- read.csv("../prev_af_ecg_opbidmc.csv")
tree_proj_complete = merge(results, bidmc_af_subset, by = "filename") 

# Merging sub-branches to derive representative branches/phenogroups using hierarchical clustering on tree coordinates
tree_proj_hierach <- tree_proj_complete %>% select(branch,Z1, Z2)
cols_to_scale <- c("Z1", "Z2")
tree_proj_hierach[, cols_to_scale] <- scale(tree_proj_hierach[, cols_to_scale])
tree_proj_hierach_grouped <- tree_proj_hierach %>%
  group_by(branch) %>%
  dplyr::summarise(across(everything(), mean))  %>% select(-branch)
rownames(tree_proj_hierach_grouped) = levels(tree_proj_hierach$branch)

dist_mat <- dist(bidmc_full_data_grouped, method = 'euclidean')
hclust_avg <- hclust(dist_mat, method = 'average')
avg_dend_obj <- as.dendrogram(hclust_avg)

avg_col_dend <- color_branches(avg_dend_obj, h = 1.2)
plot(avg_col_dend) # dendrogram colored by height to cut with 

# Plot Silhouette plot across resulting clusters
sil_cl <- silhouette(cutree(hclust_avg, h=1.2) ,dist_mat, title=title(main = 'Silhoutte Scores across Clusters at height _'))
rownames(sil_cl) = rownames(bidmc_full_data_grouped)
plot(sil_cl)

tree_proj_complete$merged_branchcoords = tree_proj_complete$branch %>%
  plyr::mapvalues(., c(1,2,3,4,5,6,7,8,25,27), c(1, 1, 1, 1, 1,1, 1, 1, 1, 1)) %>%
  plyr::mapvalues(., c(9,26), c(2,2)) %>%
  plyr::mapvalues(., c(10,11,18,19,20), c(3, 3,3, 3, 3)) %>%
  plyr::mapvalues(., c(12,13,14,15,16,17), c(4,4,4,4,4,4)) %>%
  plyr::mapvalues(., c(21,22,23,24), c(5,5,5,5))

tree_proj_complete$merged_branchcoords <- factor(tree_proj_complete$merged_branchcoords, levels = c(1,2,3,4,5))

# Visualise merged branches
tree_proj_complete$merged_branchcoords <- factor(tree_proj_complete$merged_branchcoords, 
                                                 labels = c("Branch 1", "Branch 2", "Branch 3",
                                                            "Branch 4", "Branch 5"))

gg_full <- ggplot(tree_proj_complete) +
  geom_point(aes(x=Z1, y=Z2, color=merged_branchcoords), alpha = 0.1, size = 2) + 
  scale_color_manual(values = c("firebrick3", "gold3", "forestgreen", "royalblue", "orchid4"), 
                     guide = guide_legend(override.aes = list(size = 5, alpha = 1))) + 
  labs(
    x = "Dimension 1",
    y = "Dimension 2") +
  theme_bw() +
  coord_cartesian(xlim = c(-3, 3.1), ylim = c(-3.1, 2.7)) +
  theme(
    legend.text = element_text(size = 16, face = "bold"),  # Bolden legend text
    legend.title = element_blank(),  # Bolden legend title
    axis.text = element_text(size = 14, face = "bold"),     # Bolden axis label text
    axis.title = element_text(size = 16, face = "bold"),    # Bolden x and y axis label text
    plot.title = element_text(size = 16, hjust = 0.5, face = "bold")  # Bolden plot title
  )

gg_full
